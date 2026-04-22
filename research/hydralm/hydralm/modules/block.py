"""
A HydraBlock is the pre-norm residual unit:

    y = x + mixer(norm1(x))
    y = y + mlp  (norm2(y))

The mixer is dispatched by config: either Gated DeltaNet (default) or
Sliding-Window Attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import HydraConfig
from .gated_deltanet import GatedDeltaNet
from .rmsnorm import RMSNorm
from .sliding_window import SlidingWindowAttention
from .swiglu import SwiGLU


class HydraBlock(nn.Module):
    def __init__(self, cfg: HydraConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]

        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if self.layer_type == "deltanet":
            self.mixer: nn.Module = GatedDeltaNet(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                short_conv_kernel=cfg.dn_short_conv_kernel,
                chunk_size=cfg.dn_chunk_size,
                use_gate=cfg.dn_use_gate,
                norm_qk=cfg.dn_norm_qk,
            )
        elif self.layer_type == "swa":
            self.mixer = SlidingWindowAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                window=cfg.swa_window,
                rope_base=cfg.swa_rope_base,
            )
        else:
            raise ValueError(f"Unknown layer type {self.layer_type!r}")

        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, mult=cfg.mlp_mult, multiple_of=cfg.mlp_multiple_of)

    def forward(self, x: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        h, new_state = self.mixer(self.norm1(x), state)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def step(self, x_t: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        h, new_state = self.mixer.step(self.norm1(x_t), state)
        x_t = x_t + h
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, new_state
