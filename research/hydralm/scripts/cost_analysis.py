"""
Analytic cost model: HydraLM vs a matched Transformer.

We compute *per-sequence* and *per-token* quantities from the closed-form
FLOP / memory equations — no wall-clock noise, no kernel-level surprises.
The numbers below are what a perfectly-implemented version of each model
consumes at a given (batch, seq_len, d_model, n_layers, n_heads) point.

The cost reduction reported here is *architectural* — it does not depend
on quantization, flash attention, or paged KV caches.  Those optimisations
would widen the gap further.

FLOP accounting (forward, causal, per sequence):

  Transformer block at length N with d=d_model, h=n_heads, d_h=d/h:
      attention : 4·B·N·d² + 2·B·h·N²·d_h   =   4·B·N·d²  +  2·B·N²·d
      MLP       : 16/3 · B · N · d²    (SwiGLU, 8/3·d hidden, 3 matmuls)
      total/block ≈ (4 + 16/3)·B·N·d² + 2·B·N²·d

  HydraLM DeltaNet block at length N:
      projections : 4·B·N·d²                    (qkv + out)
      chunkwise   : 2·B·N·h·d_h² + 2·B·N·h·d_h² ≈ 4·B·N·d·d_h
      gate        : B·N·d²                      (+ tiny overhead)
      MLP         : 16/3 · B · N · d²
      total/block ≈ (5 + 16/3)·B·N·d² + 4·B·N·d·d_h

  HydraLM SWA block at length N, window W:
      projections : 4·B·N·d²
      attention   : 2·B·N·min(N,W)·d
      MLP         : 16/3 · B · N · d²
      total/block ≈ (4 + 16/3)·B·N·d² + 2·B·N·min(N,W)·d

HBM-bytes (activation) accounting, per sequence, per block:

  Transformer attention KV cache (fp16): 4·B·N·d bytes
  HydraLM SWA KV cache: 4·B·min(N,W)·d bytes
  HydraLM DeltaNet state: 2·B·h·d_h² bytes (constant in N)

Dollar cost uses a reference GPU rate ($/GPU-hour) and realised TFLOP/s.
Defaults: H100 at $3.00/hr, 989 TFLOP/s FP16 effective.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class ModelSpec:
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int

    @property
    def n_dn(self) -> int: return self.n_layers - self.n_swa
    n_swa: int = 0
    swa_window: int = 512

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim


def flops_transformer(m: ModelSpec, B: int, N: int) -> float:
    # SwiGLU MLP with 8/3 · d hidden uses 3 matmuls of size B·N·d·(8/3·d) = 8/3·B·N·d² each
    # → 16/3·2 = 32/3 FMAs ≈ use 16/3 · 2 = 32/3 macs; we use `multiply-adds as 2 FLOPs`
    mlp = (32.0 / 3.0) * B * N * m.d_model ** 2
    proj = 4.0 * B * N * m.d_model ** 2                 # qkv + out
    attn = 4.0 * B * N * N * m.d_model                  # softmax attention (2 mm of N×d vs N×d)
    return m.n_layers * (proj + attn + mlp)


def flops_hydralm(m: ModelSpec, B: int, N: int) -> float:
    mlp = (32.0 / 3.0) * B * N * m.d_model ** 2
    # DeltaNet: projections + chunkwise recurrence ≈ 2·B·N·d·d_h (inter + intra chunk)
    dn_proj = 4.0 * B * N * m.d_model ** 2
    dn_core = 4.0 * B * N * m.d_model * m.head_dim
    dn_gate = 2.0 * B * N * m.d_model ** 2              # gate_proj + silu + elem prod
    dn_block = dn_proj + dn_core + dn_gate + mlp
    # SWA: projections + windowed attention
    swa_proj = 4.0 * B * N * m.d_model ** 2
    swa_attn = 4.0 * B * N * min(N, m.swa_window) * m.d_model
    swa_block = swa_proj + swa_attn + mlp
    return m.n_dn * dn_block + m.n_swa * swa_block


def hbm_bytes_transformer(m: ModelSpec, B: int, N: int, bytes_per_el: int = 2) -> float:
    # KV cache only — the dominant inference memory.
    return m.n_layers * 2.0 * B * N * m.d_model * bytes_per_el


def hbm_bytes_hydralm(m: ModelSpec, B: int, N: int, bytes_per_el: int = 2) -> float:
    dn_state = m.n_dn * 2.0 * B * m.n_heads * m.head_dim * m.head_dim * bytes_per_el
    swa_kv = m.n_swa * 2.0 * B * min(N, m.swa_window) * m.d_model * bytes_per_el
    return dn_state + swa_kv


def dollars(flops: float, gpu_flops_per_s: float, dollars_per_hour: float) -> float:
    return flops / gpu_flops_per_s / 3600.0 * dollars_per_hour


def fmt_flops(f: float) -> str:
    for unit, scale in [("E", 1e18), ("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6)]:
        if f >= scale:
            return f"{f/scale:8.2f} {unit}F"
    return f"{f:10.0f}  F"


def fmt_bytes(b: float) -> str:
    for unit, scale in [("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)]:
        if b >= scale:
            return f"{b/scale:8.3f} {unit}"
    return f"{b:6.0f}   B"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--n-swa", type=int, default=8)
    p.add_argument("--swa-window", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[2_048, 8_192, 32_768, 131_072, 1_048_576])
    p.add_argument("--gpu-tflops", type=float, default=989.0, help="effective TFLOP/s")
    p.add_argument("--gpu-dollars-per-hour", type=float, default=3.0)
    args = p.parse_args()

    hydra = ModelSpec(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, head_dim=args.head_dim,
        n_swa=args.n_swa, swa_window=args.swa_window,
    )
    xformer = ModelSpec(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, head_dim=args.head_dim,
        n_swa=0, swa_window=0,
    )
    gpu_flops = args.gpu_tflops * 1e12

    print(f"Model: d={hydra.d_model}, L={hydra.n_layers}, H={hydra.n_heads}×{hydra.head_dim}")
    print(f"  Transformer : {xformer.n_layers} softmax-attn layers")
    print(f"  HydraLM     : {hydra.n_dn} DeltaNet + {hydra.n_swa} SWA (window={hydra.swa_window})")
    print()
    print(
        f"{'N':>10}  {'Xformer FLOP':>14}  {'Hydra FLOP':>14}  "
        f"{'FLOP save':>9}  {'Xformer KV':>12}  {'Hydra state':>12}  "
        f"{'Mem save':>8}  {'Xformer $':>10}  {'Hydra $':>10}  {'$ save':>7}"
    )
    print("-" * 140)

    for N in args.lengths:
        B = args.batch
        fx = flops_transformer(xformer, B, N)
        fh = flops_hydralm(hydra, B, N)
        mx = hbm_bytes_transformer(xformer, B, N)
        mh = hbm_bytes_hydralm(hydra, B, N)
        dx = dollars(fx, gpu_flops, args.gpu_dollars_per_hour) * 1000  # per 1M toks prorate below
        dh = dollars(fh, gpu_flops, args.gpu_dollars_per_hour) * 1000
        print(
            f"{N:>10,}  {fmt_flops(fx):>14}  {fmt_flops(fh):>14}  "
            f"{(1-fh/fx)*100:>8.1f}%  {fmt_bytes(mx):>12}  {fmt_bytes(mh):>12}  "
            f"{(1-mh/mx)*100:>7.1f}%  ${dx:>9.4f}  ${dh:>9.4f}  {(1-dh/dx)*100:>6.1f}%"
        )

    print()
    print("Notes:")
    print("  * FLOP save == architectural compute reduction (excludes kernel-level gains).")
    print("  * Mem save  == inference-time state memory (KV cache vs DN state + SWA cache).")
    print("  * $ save    == compute-cost proxy at stated GPU rate.")
    print("  * Gains grow with N because Transformer cost is Θ(N²) while HydraLM is Θ(N).")


if __name__ == "__main__":
    main()
