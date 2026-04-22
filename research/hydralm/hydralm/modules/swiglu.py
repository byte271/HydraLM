"""SwiGLU feed-forward network (Shazeer, 2020).

Implementation notes
--------------------
* The gate and value projections are **fused** into a single
  ``Linear(d_model, 2 * hidden)`` weight matrix, then split.  This
  halves the number of GEMM kernel launches and allows the CUDA
  runtime to issue one larger, more efficient matmul.

* Hidden size is rounded up to the nearest ``multiple_of`` (default 64)
  so that GEMM tiles stay aligned regardless of ``d_model``.  Using
  multiples of 64 is safe for any GPU with a warp width of 32.

* No bias on the input projections (standard for LLMs).  The down
  projection also omits bias; adding one would break weight-tying
  with a matched transformer baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Fused SwiGLU: one matmul to produce gate+value, one matmul down."""

    def __init__(
        self,
        d_model: int,
        mult: float = 8 / 3,
        multiple_of: int = 64,
    ) -> None:
        super().__init__()
        # Round hidden dim up to preserve memory alignment.
        hidden = int(d_model * mult)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden

        # One fused projection for gate and value — halves kernel launches.
        self.w_fused = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul, then split along last dim.
        gv = self.w_fused(x)                          # (..., 2 * hidden)
        gate, value = gv.chunk(2, dim=-1)              # each (..., hidden)
        return self.w_down(F.silu(gate) * value)
