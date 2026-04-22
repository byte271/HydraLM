"""
Sliding-Window Attention.

A small number of these layers, interleaved with linear Gated DeltaNet
layers, recovers the precise in-context recall that pure linear-RNN
models struggle with (Samba, Jamba).

Cost: O(N * W * d_head) per layer, which is linear in N for fixed W.
With W = 512 and (say) 1 SWA per 4 layers, the global asymptote of the
full model remains dominated by the DeltaNet path.

Streaming
---------
  * ``forward`` accepts an optional prior state ``(k_cache, v_cache, pos)``
    and returns an updated state.  Chunk-wise streaming across
    arbitrarily long inputs is mathematically identical to a
    single-shot forward over the concatenated stream.
  * ``step`` decodes a single token in O(W) time and O(W) memory.

Training vs inference cache detach
-----------------------------------
During *training* the KV cache should NOT be detached so that gradients can
flow into the key/value projections when the model is trained with chunked
prefill.  During *inference* (``torch.is_grad_enabled() == False``) we detach
the cache to free the autograd graph and save memory.  The forward method
applies this policy automatically.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import RotaryEmbedding


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        window: int = 512,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim
        self.window = window
        self.scale = 1.0 / math.sqrt(head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * self.inner, bias=False)
        self.o_proj = nn.Linear(self.inner, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim, base=rope_base)

    # ---------------- parallel / prefill / streaming-chunk -----------
    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, N, _ = x.shape
        state = state or {}
        pos = state.get("pos", 0)
        k_cache = state.get("k_cache")       # (B, H, W', D) or None
        v_cache = state.get("v_cache")
        W_prev = 0 if k_cache is None else k_cache.shape[-2]

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, N, D)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, offset=pos)

        if W_prev > 0:
            k_all = torch.cat([k_cache, k], dim=-2)          # (B, H, W'+N, D)
            v_all = torch.cat([v_cache, v], dim=-2)
        else:
            k_all, v_all = k, v

        out = _windowed_causal_sdpa(
            q, k_all, v_all, window=self.window, W_prev=W_prev,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, self.inner)
        y = self.o_proj(out)

        # Keep only the last `window` rotated keys/values for the next chunk.
        # Detach when gradient is not needed (inference) to free the autograd
        # graph; keep the computation attached during training so that
        # gradients flow through the cached keys/values.
        new_k = k_all[..., -self.window:, :]
        new_v = v_all[..., -self.window:, :]
        if not torch.is_grad_enabled():
            new_k = new_k.detach()
            new_v = new_v.detach()

        new_state = {
            "k_cache": new_k,
            "v_cache": new_v,
            "pos": pos + N,
        }
        return y, new_state

    # ---------------- single-token step ------------------------------
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict | None) -> tuple[torch.Tensor, dict]:
        """Decode one token, updating the sliding-window KV cache in place."""
        B = x_t.shape[0]
        state = state or {}
        pos = state.get("pos", 0)

        qkv_t = self.qkv_proj(x_t)
        q_t, k_t, v_t = qkv_t.chunk(3, dim=-1)
        q_t = q_t.view(B, self.n_heads, 1, self.head_dim)
        k_t = k_t.view(B, self.n_heads, 1, self.head_dim)
        v_t = v_t.view(B, self.n_heads, 1, self.head_dim)
        q_t, k_t = self.rope(q_t, k_t, offset=pos)

        k_cache = state.get("k_cache")
        v_cache = state.get("v_cache")
        if k_cache is None:
            k_cache = k_t
            v_cache = v_t
        else:
            k_cache = torch.cat([k_cache, k_t], dim=-2)
            v_cache = torch.cat([v_cache, v_t], dim=-2)
            if k_cache.shape[-2] > self.window:
                k_cache = k_cache[..., -self.window:, :]
                v_cache = v_cache[..., -self.window:, :]

        # ``is_causal=False``: the query attends to the entire cache which
        # is already causally bounded (all cached keys precede or equal the
        # current token position).
        out = F.scaled_dot_product_attention(q_t, k_cache, v_cache, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, self.inner)
        y = self.o_proj(out)

        return y, {"k_cache": k_cache, "v_cache": v_cache, "pos": pos + 1}


# ---------------------------------------------------------------------------
# Windowed causal attention helper
# ---------------------------------------------------------------------------

def _windowed_causal_sdpa(
    q: torch.Tensor,       # (B, H, N, D)
    k: torch.Tensor,       # (B, H, W'+N, D)
    v: torch.Tensor,
    window: int,
    W_prev: int,
) -> torch.Tensor:
    """Sliding-window causal attention from N queries to W'+N keys.

    For query i (0 <= i < N):
        valid key j satisfies  (i + W_prev - window) < j  <=  (i + W_prev)
    which enforces both causality (no future keys) and the sliding window
    (at most ``window`` past keys).

    The fast path (no prior cache, all queries fit inside the window)
    delegates to PyTorch's built-in causal SDPA which may use FlashAttention
    on CUDA.
    """
    B, H, N, D = q.shape
    K = k.shape[-2]
    assert K == W_prev + N

    # Fast path: no prior cache and the entire sequence fits within the window.
    if W_prev == 0 and N <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    device = q.device
    # Build a (N, K) boolean validity mask.
    iq = torch.arange(N, device=device).unsqueeze(1)              # (N, 1)
    jk = torch.arange(K, device=device).unsqueeze(0)              # (1, K)
    valid = (jk > iq + W_prev - window) & (jk <= iq + W_prev)
    # Convert to an additive attention bias: 0 for valid, -inf for masked.
    bias = torch.zeros(N, K, device=device, dtype=q.dtype)
    bias.masked_fill_(~valid, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)
