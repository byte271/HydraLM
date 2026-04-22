"""Short depth-wise causal convolution (Mamba-style token shift).

Provides a tiny amount of local mixing before the linear recurrence,
which the recurrence itself cannot cheaply express.  Kernel size 4 is
the standard choice (Mamba, Mamba-2, DeltaNet).

Both forms accept an incremental cache so that chunk-wise streaming
across arbitrarily long inputs is mathematically identical to a
single-shot forward over the concatenated stream.

Performance notes
-----------------
* The parallel path (``forward``) uses ``F.conv1d`` which maps to cuDNN on
  GPU.  For the common case of depth-wise conv1d with small kernel sizes
  cuDNN uses an implicit-GEMM path that is near-optimal.
* The recurrent step (``step``) avoids a ``conv1d`` call entirely: for a
  depth-wise kernel of size K applied to a single frame, the convolution
  reduces to a dot product per channel, which we compute with a vectorised
  element-wise multiply + sum that compiles cleanly via TorchScript.
* The ``step`` path pre-caches ``conv.weight.squeeze(1)`` as a plain tensor
  attribute to avoid the overhead of the attribute lookup + squeeze on every
  decoding step.  The cache is invalidated automatically when the module is
  re-initialised (``__init__``) and kept consistent during ``load_state_dict``
  via a ``_load_from_state_dict`` hook.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    """Causal depth-wise convolution with optional streaming cache.

    Args:
        dim:         number of input/output channels (depth-wise, so groups=dim).
        kernel_size: causal receptive field (default 4, matching Mamba/DeltaNet).
    """

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            groups=dim,          # depth-wise
            padding=0,           # causal padding handled by cache / explicit pad
            bias=True,
        )
        # Pre-flattened weight cache for the O(1) step path.
        # Shape: (dim, kernel_size) — squeezed from conv.weight's (dim, 1, K).
        self._register_step_weight_cache()

    def _register_step_weight_cache(self) -> None:
        """Materialise the (dim, K) weight view used by the step path."""
        # Stored as a plain tensor (not a parameter) so it is not included in
        # ``state_dict`` and does not affect parameter counts.
        self._step_weight: torch.Tensor = self.conv.weight.squeeze(1).detach()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Keep the step weight cache in sync after a state_dict load."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._register_step_weight_cache()

    # ------------------------------------------------------------------
    # Parallel (training / prefill / streaming-chunk)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Causal depth-wise conv over a sequence chunk.

        Args:
            x:     (B, N, D) — current chunk activations.
            cache: (B, D, K-1) or None — last K-1 positions of the
                   previous chunk; only needed for streaming continuation.

        Returns:
            y:         (B, N, D) — activated conv output (SiLU applied).
            new_cache: (B, D, K-1) or None if kernel_size == 1.
        """
        b, n, d = x.shape
        k = self.kernel_size
        x_perm = x.transpose(1, 2)                              # (B, D, N)

        if cache is None:
            x_padded = F.pad(x_perm, (k - 1, 0))               # zero causal pad
        else:
            x_padded = torch.cat([cache, x_perm], dim=-1)       # (B, D, K-1+N)

        y = self.conv(x_padded)                                  # (B, D, N)
        new_cache = x_padded[..., -(k - 1):] if k > 1 else None
        return F.silu(y.transpose(1, 2)), new_cache

    # ------------------------------------------------------------------
    # Recurrent single-token step
    # ------------------------------------------------------------------
    def step(
        self,
        x_t: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one token through the causal conv — O(D * K) work.

        Args:
            x_t:   (B, D) — single token activations.
            cache: (B, D, K-1) or None — rolling input window.

        Returns:
            y_t:   (B, D) — SiLU-activated output.
            cache: (B, D, K-1) — updated rolling window (shift-left by 1).
        """
        b, d = x_t.shape
        k = self.kernel_size
        if cache is None:
            cache = x_t.new_zeros(b, d, k - 1)

        # window: (B, D, K) = [cache | x_t]
        window = torch.cat([cache, x_t.unsqueeze(-1)], dim=-1)

        # Depth-wise dot product: sum over K for each channel independently.
        # self._step_weight: (D, K)  →  broadcast to (1, D, K)
        weight = self._step_weight.to(x_t.device)               # handle device moves
        y_t = (window * weight.unsqueeze(0)).sum(dim=-1)         # (B, D)
        y_t = y_t + self.conv.bias                               # (B, D)

        new_cache = window[..., 1:]                              # (B, D, K-1)
        return F.silu(y_t), new_cache
