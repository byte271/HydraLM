"""
Dense causal Transformer reference baseline.

This is the model HydraLM claims to replace.  Its job in this repo is to
make every comparison (MQAR recall, length scaling, cost analysis) an
apples-to-apples head-to-head:

  * Same HydraConfig hyperparameters (``d_model``, ``n_layers``,
    ``n_heads``, ``head_dim``, ``vocab_size``).
  * Same I/O contract: ``model(input_ids) -> {"logits": ...}``.
  * Same SwiGLU MLP with ``mlp_mult`` / ``mlp_multiple_of`` from config
    and the same RMSNorm pre-norm, so parameter counts match to <0.1%.
  * RoPE positional encoding, matching the SWA layers in HydraLM.

What it is NOT:
  * FlashAttention — we use PyTorch's built-in SDPA so the kernel choice
    is dictated by hardware, not by us.
  * Sliding-window or sparse — this is the dense O(N^2) reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydralm.config import HydraConfig
from hydralm.modules.rmsnorm import RMSNorm
from hydralm.modules.rotary import RotaryEmbedding
from hydralm.modules.swiglu import SwiGLU


class CausalSelfAttention(nn.Module):
    """Standard dense causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * self.inner, bias=False)
        self.o_proj = nn.Linear(self.inner, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, offset=0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner)
        return self.o_proj(out)


class DenseBlock(nn.Module):
    """Pre-norm SwiGLU transformer block."""

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.head_dim, rope_base=cfg.swa_rope_base,
        )
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, mult=cfg.mlp_mult,
                          multiple_of=cfg.mlp_multiple_of)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DenseTransformer(nn.Module):
    """A dense O(N^2) Transformer that matches HydraLM's I/O contract.

    The module takes the **same** ``HydraConfig`` as HydraLM, so any script
    constructed against ``HydraLM(cfg)`` can swap in ``DenseTransformer(cfg)``
    without touching a single hyperparameter.
    """

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie input / output embeddings — matches HydraLM default.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Use per-module dispatch so RMSNorm weights stay at their ones-init
        # default. Zeroing every 1-D parameter (including norm scales) turns
        # every pre-norm into an all-zero output and the baseline cannot
        # learn — this was the canonical "transformer at chance" bug.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        # RMSNorm intentionally untouched — its constructor sets weight=1.

    # -------- HydraLM-compatible interface ------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict:
        """Returns ``{"logits": Tensor[B, N, V]}``.

        Extra kwargs are silently dropped so the signature tolerates the
        superset of options accepted by ``HydraLM.forward`` (``state``,
        ``return_state``).
        """
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"logits": self.lm_head(x)}


__all__ = ["DenseTransformer", "DenseBlock", "CausalSelfAttention"]
