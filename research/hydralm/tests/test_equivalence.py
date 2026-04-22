"""
The three delta-rule kernels — reference, recurrent, chunkwise — must
produce numerically identical outputs (up to fp rounding).  The
`step()` path of GatedDeltaNet must match its parallel `forward()`.

If these tests pass, the chunkwise training kernel and the O(1)
inference kernel are proven equivalent, so training and deployment use
the SAME function.
"""
from __future__ import annotations

import pytest
import torch

from hydralm.kernels import (
    delta_rule_chunkwise,
    delta_rule_recurrent,
    delta_rule_reference,
)
from hydralm.modules import GatedDeltaNet


def _random_inputs(B=2, H=4, N=48, D=16, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g, dtype=dtype)
    k = torch.randn(B, H, N, D, generator=g, dtype=dtype)
    v = torch.randn(B, H, N, D, generator=g, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, N, generator=g, dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, H, N, generator=g, dtype=dtype))
    # L2 normalise q, k (as the layer does)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    return q, k, v, alpha, beta


def test_reference_vs_recurrent():
    q, k, v, a, b = _random_inputs()
    o1, S1 = delta_rule_reference(q, k, v, a, b)
    o2, S2 = delta_rule_recurrent(q, k, v, a, b)
    assert torch.allclose(o1, o2, atol=1e-10), (o1 - o2).abs().max()
    assert torch.allclose(S1, S2, atol=1e-10)


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16, 48])
def test_reference_vs_chunkwise(chunk_size):
    q, k, v, a, b = _random_inputs(N=48)
    o1, S1 = delta_rule_reference(q, k, v, a, b)
    o2, S2 = delta_rule_chunkwise(q, k, v, a, b, chunk_size=chunk_size)
    assert torch.allclose(o1, o2, atol=1e-8), (o1 - o2).abs().max()
    assert torch.allclose(S1, S2, atol=1e-8)


def test_chunkwise_non_multiple_length():
    # N not a multiple of chunk_size: padding path must still match.
    q, k, v, a, b = _random_inputs(N=37)
    o1, _ = delta_rule_reference(q, k, v, a, b)
    o2, _ = delta_rule_chunkwise(q, k, v, a, b, chunk_size=8)
    assert torch.allclose(o1, o2, atol=1e-8)


def test_state_carry_streaming():
    """Feeding a sequence in two halves with state-carry must equal the full sequence."""
    q, k, v, a, b = _random_inputs(N=40)
    o_full, S_full = delta_rule_reference(q, k, v, a, b)

    o_a, S_a = delta_rule_reference(q[:, :, :16], k[:, :, :16], v[:, :, :16],
                                    a[:, :, :16], b[:, :, :16])
    o_b, S_b = delta_rule_reference(q[:, :, 16:], k[:, :, 16:], v[:, :, 16:],
                                    a[:, :, 16:], b[:, :, 16:],
                                    initial_state=S_a)
    stitched = torch.cat([o_a, o_b], dim=2)
    assert torch.allclose(stitched, o_full, atol=1e-10)
    assert torch.allclose(S_b, S_full, atol=1e-10)


def test_deltanet_parallel_vs_step():
    """Forward(x) on a full sequence must equal calling step() one token at a time."""
    torch.manual_seed(0)
    B, N, D = 2, 24, 64
    layer = GatedDeltaNet(d_model=D, n_heads=4, head_dim=16, chunk_size=8).double()
    x = torch.randn(B, N, D, dtype=torch.float64)

    y_par, _ = layer(x)

    state = None
    step_outputs = []
    for t in range(N):
        y_t, state = layer.step(x[:, t, :], state)
        step_outputs.append(y_t)
    y_step = torch.stack(step_outputs, dim=1)

    diff = (y_par - y_step).abs().max().item()
    assert diff < 1e-7, f"parallel/step mismatch: {diff}"
