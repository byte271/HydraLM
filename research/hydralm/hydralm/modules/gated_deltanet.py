"""
Gated DeltaNet mixer.

    x  -->  short-conv  -->  [q, k, v, alpha, beta, gate]  -->  delta-rule  -->  output

Key design choices, following Yang et al. (2024) "Gated DeltaNet":

  * q and k are L2-normalised per head.  Without this, the state matrix
    grows unboundedly and training diverges; with it, the induced
    kernel is a cosine similarity and the delta rule behaves like a
    corrective associative memory.

  * alpha_t (forget gate) = sigmoid(linear(x_t))^{1/tau} with tau >= 1.
    This keeps alpha very close to 1 by default so long-range
    dependencies are preserved unless the input explicitly demands
    forgetting.

  * beta_t (write strength) = sigmoid(linear(x_t)).

  * An output gate (SiLU) is applied elementwise; this is what makes
    the layer "gated" in the same sense as Mamba / GLA.

During training we call `delta_rule_chunkwise`; during single-token
generation we call a streaming step that maintains:
    - the short-conv rolling cache (B, D, K-1)
    - the recurrent state S       (B, H, Dv, Dk)

Implementation notes
--------------------
* alpha and beta are **fused** into a single Linear(d_model, 2*n_heads)
  so that both are computed in one GEMM.

* The two dead helper methods (_project, _split_heads) that were
  defined but never called in the original have been removed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.delta_rule import delta_rule_chunkwise
from .short_conv import ShortConv


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        short_conv_kernel: int = 4,
        chunk_size: int = 64,
        use_gate: bool = True,
        norm_qk: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim
        self.chunk_size = chunk_size
        self.use_gate = use_gate
        self.norm_qk = norm_qk

        # Fused q / k / v projection — one GEMM.
        self.qkv_proj = nn.Linear(d_model, 3 * self.inner, bias=False)

        # Fused alpha + beta projection — one GEMM instead of two.
        # Output layout: [..., :n_heads] = alpha logits, [..., n_heads:] = beta logits.
        self.ab_proj = nn.Linear(d_model, 2 * n_heads, bias=True)

        # Mamba-style pre-conv on the concatenated q,k,v path for token shift.
        self.short_conv = ShortConv(3 * self.inner, kernel_size=short_conv_kernel)

        if use_gate:
            self.gate_proj = nn.Linear(d_model, self.inner, bias=False)

        self.o_proj = nn.Linear(self.inner, d_model, bias=False)

        # Initialise alpha bias so sigmoid(bias) ~ 0.98, i.e. very mild decay.
        # Beta bias stays zero (neutral write strength at init).
        with torch.no_grad():
            self.ab_proj.bias[:n_heads].fill_(math.log(0.98 / 0.02))
            self.ab_proj.bias[n_heads:].zero_()

    # =================================================================
    #  Parallel (training / prefill) forward
    # =================================================================
    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x:     (B, N, d_model)
            state: optional dict with keys 'conv_cache', 'S'
        Returns:
            y:     (B, N, d_model)
            state: updated state dict (for streaming)
        """
        B, N, _ = x.shape
        state = state or {}

        qkv = self.qkv_proj(x)                               # (B, N, 3*inner)
        qkv, conv_cache = self.short_conv(qkv, state.get("conv_cache"))
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, H, N, D) for the kernel.
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.norm_qk:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # Fused alpha + beta in one call.
        ab = self.ab_proj(x)                                  # (B, N, 2*H)
        alpha = torch.sigmoid(ab[..., :self.n_heads]).transpose(1, 2)   # (B, H, N)
        beta  = torch.sigmoid(ab[..., self.n_heads:]).transpose(1, 2)   # (B, H, N)

        out, S = delta_rule_chunkwise(
            q, k, v, alpha, beta,
            chunk_size=self.chunk_size,
            initial_state=state.get("S"),
        )                                                    # out: (B, H, N, D)

        out = out.transpose(1, 2).contiguous().view(B, N, self.inner)

        if self.use_gate:
            g = F.silu(self.gate_proj(x))
            out = out * g

        out = self.o_proj(out)

        return out, {"S": S, "conv_cache": conv_cache}

    # =================================================================
    #  Recurrent single-token step (O(1) w.r.t. context length)
    # =================================================================
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict | None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x_t:   (B, d_model)
            state: dict with optional keys:
                     - 'conv_cache': (B, 3*inner, K-1)
                     - 'S'         : (B, H, D, D)
        """
        B = x_t.shape[0]
        state = state or {}

        qkv_t = self.qkv_proj(x_t)                           # (B, 3*inner)
        qkv_t, conv_cache = self.short_conv.step(qkv_t, state.get("conv_cache"))
        q_t, k_t, v_t = qkv_t.chunk(3, dim=-1)

        q_t = q_t.view(B, self.n_heads, self.head_dim)
        k_t = k_t.view(B, self.n_heads, self.head_dim)
        v_t = v_t.view(B, self.n_heads, self.head_dim)

        if self.norm_qk:
            q_t = F.normalize(q_t, p=2, dim=-1)
            k_t = F.normalize(k_t, p=2, dim=-1)

        # Fused alpha + beta.
        ab_t = self.ab_proj(x_t)                              # (B, 2*H)
        alpha_t = torch.sigmoid(ab_t[:, :self.n_heads])       # (B, H)
        beta_t  = torch.sigmoid(ab_t[:, self.n_heads:])       # (B, H)

        # One-step delta-rule recurrence (no kernel launch overhead).
        S = state.get("S")
        if S is None:
            S = x_t.new_zeros(B, self.n_heads, self.head_dim, self.head_dim)

        # a : (B, H, 1, 1) — broadcast over (D_v, D_k) dims of S
        a = alpha_t.unsqueeze(-1).unsqueeze(-1)
        b = beta_t.unsqueeze(-1)                              # (B, H, 1)

        # Retrieve current association: Sk = S k_t, shape (B, H, D)
        Sk = (S * k_t.unsqueeze(-2)).sum(-1)

        # Error-corrected write vector: u = beta * (v - alpha * Sk)
        u_t = b * (v_t - alpha_t.unsqueeze(-1) * Sk)         # (B, H, D)

        # State update: S <- alpha * S + u_t k_t^T
        S = a * S + u_t.unsqueeze(-1) * k_t.unsqueeze(-2)    # (B, H, D, D)

        # Read: o_t = S q_t
        o_t = (S * q_t.unsqueeze(-2)).sum(-1)                 # (B, H, D)
        o_t = o_t.reshape(B, self.inner)

        if self.use_gate:
            g_t = F.silu(self.gate_proj(x_t))
            o_t = o_t * g_t

        y_t = self.o_proj(o_t)
        return y_t, {"conv_cache": conv_cache, "S": S}
