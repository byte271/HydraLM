"""Rotary positional embedding (Su et al., 2021).

Used only inside sliding-window attention layers.  Gated DeltaNet does
not use positional embeddings — the recurrence supplies ordering
information directly, which is part of why it extrapolates to
arbitrarily long contexts.

Implementation notes
--------------------
* The cosine/sine tables are computed lazily and cached with a bounded
  LRU policy: we keep at most ``MAX_CACHED_SEQLEN`` positions to avoid
  unbounded memory growth during million-token streaming.
* A single ``(seqlen, device, dtype)`` key is used so the same RotaryEmbedding
  instance can be used on multiple devices (e.g. CPU for tests, GPU for
  training) without cross-device tensor errors.
* ``head_dim`` must be even (asserted at construction).
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn


# Maximum number of (cos, sin) slabs to keep in the per-instance cache.
# Each slab is (seqlen, head_dim) fp32 → 2 × seqlen × head_dim floats.
# At head_dim=128 and seqlen=32768, one slab ≈ 32 MiB; capping at 8 slabs
# is conservative for all architectures in this codebase.
_MAX_CACHED_SLABS = 8


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with bounded cosine/sine cache.

    Args:
        head_dim: dimensionality of each attention head. Must be even.
        base:     RoPE base frequency (default 10 000).
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Bounded cache: maps (seqlen, device_str, dtype) -> (cos, sin).
        # Using an OrderedDict with manual eviction to cap size.
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_order: list[tuple] = []

    # ------------------------------------------------------------------
    def _cos_sin(
        self,
        seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for ``seqlen`` positions.

        Results are cached with LRU eviction at ``_MAX_CACHED_SLABS`` entries
        to bound memory regardless of streaming sequence length.
        """
        key = (seqlen, str(device), dtype)
        if key in self._cache:
            # Move to end (most-recently-used).
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        # Compute new slab.
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.einsum("n,d->nd", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)              # (seqlen, head_dim)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        # Evict oldest entry if cache is full.
        if len(self._cache_order) >= _MAX_CACHED_SLABS:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (cos, sin)
        self._cache_order.append(key)
        return cos, sin

    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,          # (B, H, N, D)
        k: torch.Tensor,          # (B, H, N, D)
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to ``q`` and ``k``.

        Args:
            q, k:   Query and key tensors of shape (B, H, N, D).
            offset: Absolute position index of the first token in this chunk.
                    Pass the running ``pos`` counter from the SWA layer cache
                    to correctly handle streaming chunk-wise prefill.

        Returns:
            (rotated_q, rotated_k) with the same shapes as the inputs.
        """
        n = q.shape[-2]
        cos, sin = self._cos_sin(offset + n, q.device, q.dtype)
        # Slice out the ``n`` positions for this chunk.
        cos = cos[offset:offset + n]   # (N, D)
        sin = sin[offset:offset + n]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


# ---------------------------------------------------------------------------
# Low-level rotation kernel
# ---------------------------------------------------------------------------

def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` using the precomputed ``cos`` / ``sin`` tables.

    The standard RoPE rotation interleaves pairs (x[2i], x[2i+1]) as
    complex numbers.  This implementation uses the equivalent "split-half"
    form which avoids reshaping and is just as numerically stable:

        rotate(x) = x * cos + [-x2, x1] * sin

    where x1 = x[..., :D//2] and x2 = x[..., D//2:].

    Args:
        x:   (B, H, N, D) or any tensor whose last dim is the head dim.
        cos: (N, D) — broadcast over B and H.
        sin: (N, D) — broadcast over B and H.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    # Unsqueeze cos/sin for broadcasting over (B, H).
    cos = cos.unsqueeze(0).unsqueeze(0)    # (1, 1, N, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotated * sin
