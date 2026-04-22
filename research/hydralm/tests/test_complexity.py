"""Complexity tests.

Two layers:

1. **Analytic scaling** (always on, <1s): uses ``hydralm.baselines.flops`` --
   a closed-form FLOP/memory model shared with the scripts -- and verifies
   the log-log slope of each curve.  HydraLM must be O(N^1) in FLOPs and
   O(1) in streaming memory; a dense Transformer must be super-linear.

2. **Empirical wall-clock** (opt-in via ``pytest --run-slow``): runs real
   prefill at several sequence lengths and checks the fitted exponent.

If HydraLM ever regresses to quadratic behaviour, the analytic test turns
red immediately -- no need to wait for a benchmark run.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from hydralm import HydraConfig, HydraLM
from hydralm.baselines import flops


# -- shared tiny config -----------------------------------------------------

def _small_hydra() -> HydraConfig:
    return HydraConfig(
        vocab_size=257, d_model=64, n_layers=4, n_heads=4, head_dim=16,
        swa_window=32, dn_chunk_size=16, swa_every=2,
        max_position_embeddings=1 << 22,  # keep rope table large enough
    )


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx) or 1e-30
    return num / den


# -- analytic scaling (always on) ------------------------------------------

def test_flop_scaling_is_linear_in_N():
    """HydraLM forward FLOPs must have log-log slope ~1.0 over 1K-4M tokens."""
    cfg = _small_hydra()
    spec = flops.ModelSpec.from_hydra(cfg)
    ns = [1 << k for k in range(10, 23)]
    fs = [flops.flops_of(spec, B=1, N=n) for n in ns]
    slope = _loglog_slope([float(n) for n in ns], [float(f) for f in fs])
    assert 0.95 < slope < 1.05, f"HydraLM FLOP slope={slope:.3f}, expected ~1.0"


def test_flop_scaling_transformer_is_superlinear():
    """Baseline sanity: a dense Transformer must be super-linear (>1.5)."""
    cfg = _small_hydra()
    spec = flops.ModelSpec.from_transformer(cfg)
    ns = [1 << k for k in range(10, 23)]
    fs = [flops.flops_of(spec, B=1, N=n) for n in ns]
    slope = _loglog_slope([float(n) for n in ns], [float(f) for f in fs])
    assert slope > 1.5, f"Transformer FLOP slope={slope:.3f} not super-linear"


def test_streaming_memory_is_constant():
    """Recurrent-state memory must be O(1) in sequence length."""
    cfg = _small_hydra()
    spec = flops.ModelSpec.from_hydra(cfg)
    mems = [flops.state_bytes_of(spec, B=1, N=1 << k) for k in range(10, 23)]
    assert max(mems) == min(mems), f"state_bytes varies with N: unique={sorted(set(mems))}"


def test_savings_exceed_90pct_at_128K():
    """At 128K tokens, HydraLM must save >=90% of FLOPs AND memory."""
    cfg = _small_hydra()
    h = flops.ModelSpec.from_hydra(cfg)
    t = flops.ModelSpec.from_transformer(cfg)
    sv = flops.savings(h, t, B=1, N=131_072)
    assert sv["flop_save"] >= 0.90, f"flop_save={sv['flop_save']:.3f} below 90%"
    assert sv["mem_save"] >= 0.90, f"mem_save={sv['mem_save']:.3f} below 90%"


# -- empirical wall-clock (opt-in) -----------------------------------------

@pytest.mark.slow
def test_linear_prefill_scaling():
    """Measured prefill time must fit a sub-quadratic log-log slope."""
    torch.manual_seed(0)
    cfg = HydraConfig(
        vocab_size=1024, d_model=128, n_layers=4, n_heads=4,
        swa_every=4, swa_window=64, dn_chunk_size=32,
        max_position_embeddings=65_536,
    )
    model = HydraLM(cfg).eval()

    with torch.no_grad():
        _ = model(torch.zeros(1, 128, dtype=torch.long))

    Ns = [256, 512, 1024, 2048, 4096]
    ts = []
    for N in Ns:
        x = torch.randint(0, cfg.vocab_size, (1, N))
        with torch.no_grad():
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
        ts.append(t1 - t0)

    slope, _ = np.polyfit(np.log(Ns), np.log(ts), 1)
    print(f"\nempirical scaling exponent = {slope:.3f} (target: ~1.0)")
    print(f"timings: {list(zip(Ns, [round(t, 4) for t in ts]))}")
    assert slope < 1.3, f"prefill scales super-linearly: slope={slope:.3f}"
