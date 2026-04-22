"""
Million-token streaming demo.

Runs HydraLM over 1,000,000 tokens on CPU and prints a live state-memory
trace.  The trace shows peak-state-bytes across every chunk — this number
must be constant after the first chunk, regardless of total length.

Total wall time on a laptop CPU for a tiny debug model (d=64, L=2):
  ~20–60 seconds.  Memory: <10 MiB of state across the whole run.

This is the concrete empirical demonstration that the architecture is
sub-quadratic in more than just the asymptotic sense.
"""
from __future__ import annotations

import argparse
import time

import torch

from hydralm import HydraConfig, HydraLM
from hydralm.streaming import StreamingEngine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--swa-window", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    cfg = HydraConfig(
        vocab_size=512,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        swa_window=args.swa_window,
        swa_every=2,
        dn_chunk_size=64,
    )
    model = HydraLM(cfg)
    n_params = model.num_parameters()

    print(f"Model: {cfg.summary()}")
    print(f"  params: {n_params:,}")
    print(f"  chunk size: {args.chunk_size:,} tokens")
    print(f"  total to process: {args.tokens:,} tokens "
          f"({args.tokens / args.chunk_size:,.0f} chunks)")
    print()

    engine = StreamingEngine(model, chunk_size=args.chunk_size, dtype=torch.float32)

    # Synthetic token stream — any data would do; we're measuring compute + state.
    tokens = torch.randint(0, cfg.vocab_size, (1, args.tokens))

    t0 = time.perf_counter()
    stats = engine.process(tokens, progress_every=max(1, args.tokens // args.chunk_size // 20))
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 72)
    print(f"DONE   {stats.summary()}")
    print(f"       wall time: {elapsed:.2f} s "
          f"({stats.tokens_processed / elapsed:,.1f} tok/s)")
    print(f"       peak state memory: {stats.peak_state_bytes / 1024**2:.3f} MiB")
    print(f"       memory per token:  {stats.peak_state_bytes / stats.tokens_processed:.6f} B/tok")
    print("=" * 72)
    print()
    print("Interpretation: peak state memory is independent of the total")
    print("token count (re-run with --tokens 10000 and compare).  This is")
    print("the architectural property that makes 1M-10M-token contexts")
    print("feasible without a datacentre.")


if __name__ == "__main__":
    main()
