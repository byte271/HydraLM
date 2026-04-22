"""
HydraLM configuration.

The architecture is a stack of N blocks. Each block is one of:

  * "deltanet" : Gated DeltaNet linear-recurrent mixer   (O(N) time, O(1) state per token)
  * "swa"      : Sliding-Window Attention                (O(N * W) time, O(W) state per token)

Hybridisation follows Samba / Jamba: a small fraction of SWA layers is
sufficient to recover precise in-context recall, while the bulk of the
compute remains linear.  The default schedule is:

    [DN, DN, DN, SWA, DN, DN, DN, SWA, ...]   (1 SWA per 4 layers)

which keeps > 85% of FLOPs in the linear path at every sequence length
and yields a strict O(N * (d_model^2 + W * d_model)) asymptotic cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

LayerType = Literal["deltanet", "swa"]


@dataclass
class HydraConfig:
    # --- vocabulary ---
    vocab_size: int = 32_000
    pad_token_id: int = 0

    # --- shape ---
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12              # used by both DN and SWA
    head_dim: int | None = None    # defaults to d_model // n_heads

    # --- MLP ---
    mlp_mult: float = 8 / 3        # SwiGLU: hidden = round(d_model * 8/3)
    mlp_multiple_of: int = 64

    # --- Gated DeltaNet ---
    dn_short_conv_kernel: int = 4  # Mamba-style token-shift conv
    dn_use_gate: bool = True       # per-token output gate
    dn_chunk_size: int = 64        # chunkwise-parallel chunk length
    dn_norm_qk: bool = True        # L2-normalise q, k (stabilises delta rule)

    # --- Sliding Window Attention ---
    swa_window: int = 512
    swa_rope_base: float = 10_000.0

    # --- block schedule ---
    # If None, auto-build: one SWA every `swa_every` layers, rest DeltaNet.
    layer_types: Sequence[LayerType] | None = None
    swa_every: int = 4

    # --- misc ---
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    max_position_embeddings: int = 1_048_576  # 1M — hybrid handles this natively

    def __post_init__(self) -> None:
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0
            self.head_dim = self.d_model // self.n_heads
        if self.layer_types is None:
            self.layer_types = tuple(
                "swa" if (i + 1) % self.swa_every == 0 else "deltanet"
                for i in range(self.n_layers)
            )
        assert len(self.layer_types) == self.n_layers

    # ---------------------------------------------------------------
    @property
    def n_swa_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "swa")

    @property
    def n_dn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "deltanet")

    def summary(self) -> str:
        return (
            f"HydraConfig(d_model={self.d_model}, n_layers={self.n_layers}, "
            f"heads={self.n_heads}x{self.head_dim}, "
            f"DN={self.n_dn_layers}, SWA={self.n_swa_layers} @ window={self.swa_window})"
        )

    def to_dict(self) -> dict:
        """JSON-serialisable config dict (used by ``save_pretrained``).

        ``layer_types`` is stored as a ``list`` so that the round-trip
        through JSON is a no-op; every other field is already a plain
        Python scalar or ``None``.
        """
        from dataclasses import asdict
        d = asdict(self)
        if d.get("layer_types") is not None:
            d["layer_types"] = list(d["layer_types"])
        return d
