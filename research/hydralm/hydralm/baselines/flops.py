"""
Closed-form FLOP / memory / dollar model for Transformer vs HydraLM.

One source of truth, shared by:

  * ``scripts/cost_analysis.py``          (human-readable table)
  * ``scripts/reproduce_claims.py``       (JSON artifact)
  * ``tests/test_claims.py``              (numerical gating of "90% cost
                                           reduction" at canonical N)

Per-sequence forward FLOP accounting (B = batch, N = seq len, d = d_model,
h = n_heads, d_h = head_dim, W = SWA window):

  Transformer block:
      attention : 4·B·N·d²  +  2·B·N²·d
      MLP       : 32/3 · B·N·d²       (SwiGLU: 3 mm of shape B·N·d·(8d/3))
      total     : (4 + 32/3)·B·N·d² + 2·B·N²·d        # dominated by N²

  HydraLM DeltaNet block:
      projections : 4·B·N·d²
      chunkwise   : 4·B·N·d·d_h        # inter-chunk + intra-chunk
      gate        : 2·B·N·d²
      MLP         : 32/3 · B·N·d²
      total       : (6 + 32/3)·B·N·d² + 4·B·N·d·d_h   # linear in N

  HydraLM SWA block:
      projections : 4·B·N·d²
      attention   : 4·B·N·min(N, W)·d
      MLP         : 32/3 · B·N·d²
      total       : (4 + 32/3)·B·N·d² + 4·B·N·min(N,W)·d   # linear in N

Inference-time state memory (bytes, fp16):

  Transformer KV cache : 2 · n_layers · B · N · d · 2
  HydraLM DN state     : n_dn · 2 · B · h · d_h² · 2   # CONSTANT in N
  HydraLM SWA cache    : n_swa · 2 · B · min(N,W) · d · 2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelSpec:
    """Minimal architectural spec sufficient to compute analytical costs.

    Construct with ``ModelSpec.from_hydra(cfg)`` or ``from_transformer(cfg)``.
    """
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    n_swa: int = 0             # 0 for a pure Transformer baseline
    swa_window: int = 512
    kind: Literal["hydralm", "transformer"] = "hydralm"

    @property
    def n_dn(self) -> int:
        return self.n_layers - self.n_swa

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads*head_dim "
            f"({self.n_heads}*{self.head_dim})"
        )

    @classmethod
    def from_hydra(cls, cfg) -> "ModelSpec":
        return cls(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, head_dim=cfg.head_dim,
            n_swa=cfg.n_swa_layers, swa_window=cfg.swa_window,
            kind="hydralm",
        )

    @classmethod
    def from_transformer(cls, cfg) -> "ModelSpec":
        return cls(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, head_dim=cfg.head_dim,
            n_swa=0, swa_window=0,
            kind="transformer",
        )


# ------------------------------------------------------------------ FLOPs
def flops_transformer(m: ModelSpec, B: int, N: int) -> float:
    """Forward FLOPs of a dense causal Transformer."""
    d = m.d_model
    mlp = (32.0 / 3.0) * B * N * d * d
    proj = 4.0 * B * N * d * d
    attn = 4.0 * B * N * N * d                         # Θ(N²)
    return m.n_layers * (proj + attn + mlp)


def flops_hydralm(m: ModelSpec, B: int, N: int) -> float:
    """Forward FLOPs of HydraLM (n_dn DeltaNet + n_swa SWA blocks)."""
    d = m.d_model
    d_h = m.head_dim
    mlp = (32.0 / 3.0) * B * N * d * d

    dn_proj = 4.0 * B * N * d * d
    dn_core = 4.0 * B * N * d * d_h                    # Θ(N)
    dn_gate = 2.0 * B * N * d * d
    dn_block = dn_proj + dn_core + dn_gate + mlp

    swa_proj = 4.0 * B * N * d * d
    swa_attn = 4.0 * B * N * min(N, m.swa_window) * d  # Θ(N) if N>=W
    swa_block = swa_proj + swa_attn + mlp

    return m.n_dn * dn_block + m.n_swa * swa_block


def flops_of(m: ModelSpec, B: int, N: int) -> float:
    return flops_transformer(m, B, N) if m.kind == "transformer" \
        else flops_hydralm(m, B, N)


# ------------------------------------------------------------------ memory
def state_bytes_transformer(m: ModelSpec, B: int, N: int,
                            bytes_per_el: int = 2) -> float:
    """Inference KV-cache bytes (fp16 by default)."""
    return m.n_layers * 2.0 * B * N * m.d_model * bytes_per_el


def state_bytes_hydralm(m: ModelSpec, B: int, N: int,
                        bytes_per_el: int = 2) -> float:
    """Inference state bytes: DN state (constant in N) + SWA KV cache (bounded).

    DeltaNet state:
        Each DN layer keeps ONE matrix S ∈ R^{H × d_h × d_h} per batch item.
        There is no "K" and "V" split — the delta rule encodes both into S.
        Memory: n_dn × B × H × d_h² × bytes_per_el.

    SWA state:
        Each SWA layer keeps a sliding-window KV cache of shape
        (B, H, min(N, W), d_h) for K and again for V: factor of 2.
        Memory: n_swa × 2 × B × min(N, W) × d_model × bytes_per_el.

    The factor of 2 applies to SWA (has K and V) but NOT to DeltaNet
    (has only S).  A previous version mistakenly applied 2× to the DN
    state, inflating the reported savings by ~2×.
    """
    dn_state = m.n_dn * B * m.n_heads * m.head_dim * m.head_dim * bytes_per_el
    swa_kv = m.n_swa * 2.0 * B * min(N, m.swa_window) * m.d_model * bytes_per_el
    return dn_state + swa_kv


def state_bytes_of(m: ModelSpec, B: int, N: int,
                   bytes_per_el: int = 2) -> float:
    return state_bytes_transformer(m, B, N, bytes_per_el) \
        if m.kind == "transformer" \
        else state_bytes_hydralm(m, B, N, bytes_per_el)


# ------------------------------------------------------------------ dollars
def dollars(flops: float, gpu_flops_per_s: float,
            dollars_per_hour: float) -> float:
    return flops / gpu_flops_per_s / 3600.0 * dollars_per_hour


# ------------------------------------------------------------------ gating
def savings(
    hydra: ModelSpec, transformer: ModelSpec, B: int, N: int,
) -> dict[str, float]:
    """Return the savings triplet: compute, memory, dollar (all in [0, 1))."""
    fx = flops_transformer(transformer, B, N)
    fh = flops_hydralm(hydra, B, N)
    mx = state_bytes_transformer(transformer, B, N)
    mh = state_bytes_hydralm(hydra, B, N)
    return {
        "flop_transformer": fx,
        "flop_hydralm": fh,
        "flop_save": 1.0 - fh / fx,
        "mem_transformer": mx,
        "mem_hydralm": mh,
        "mem_save": 1.0 - mh / mx,
        "dollar_save": 1.0 - fh / fx,   # same ratio as FLOPs at fixed MFU
    }


__all__ = [
    "ModelSpec", "flops_transformer", "flops_hydralm", "flops_of",
    "state_bytes_transformer", "state_bytes_hydralm", "state_bytes_of",
    "dollars", "savings",
]
