"""
Run the MQAR benchmark to completion and save the learning curve.

This is the headline recall-capability experiment.

  python scripts/run_mqar.py \
      --d-model 128 --n-layers 4 --n-heads 4 \
      --kv-pairs 16 --queries 16 --steps 1200 \
      --out mqar_results.json

The expected outcome, per the Zoology paper (Arora et al. 2024), is:

  * Pure linear attention / Mamba-1 : stuck below 20% accuracy.
  * Gated DeltaNet                  : > 95% accuracy at these settings.
  * Hybrid (HydraLM default)        : > 98% accuracy, matches full attention.

We run HydraLM with three architecture variants side by side so the result
is an architecture-ablation, not a single number:

  1. ``pure_dn``   : all DeltaNet layers, no SWA.
  2. ``hybrid``    : default 1 SWA per 4 layers (HydraLM default).
  3. ``all_swa``   : all SWA layers (upper bound ≈ full attention).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hydralm import HydraConfig, HydraLM
from hydralm.baselines import DenseTransformer
from hydralm.eval import MQARConfig, train_mqar


def build_model(variant: str, args, vocab: int):
    """Build a model for the given variant.

    Variants
    --------
    - ``pure_dn``   : all Gated DeltaNet layers (linear attention).
    - ``hybrid``    : HydraLM default (1 SWA per ``swa_every`` layers).
    - ``all_swa``   : all SWA layers (local attention, HydraLM plumbing).
    - ``transformer``: dense causal Transformer baseline (full attention).
                      Same d_model/n_layers/n_heads and param count as the
                      HydraLM variants -- the upper bound for recall.
    """
    if variant == "pure_dn":
        layer_types = tuple("deltanet" for _ in range(args.n_layers))
    elif variant == "hybrid":
        layer_types = None  # default schedule
    elif variant == "all_swa":
        layer_types = tuple("swa" for _ in range(args.n_layers))
    elif variant == "transformer":
        layer_types = None  # irrelevant for DenseTransformer
    else:
        raise ValueError(variant)

    cfg = HydraConfig(
        vocab_size=vocab,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        swa_window=min(args.swa_window, 2 * (args.kv_pairs + args.queries) + 8),
        dn_chunk_size=args.dn_chunk_size,
        layer_types=layer_types,
        swa_every=args.swa_every,
        max_position_embeddings=4096,
    )
    if variant == "transformer":
        return DenseTransformer(cfg)
    return HydraLM(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dn-chunk-size", type=int, default=32)
    ap.add_argument("--swa-every", type=int, default=2)
    ap.add_argument("--swa-window", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--kv-pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", nargs="+",
                    default=["pure_dn", "hybrid", "all_swa", "transformer"])
    ap.add_argument("--out", type=Path, default=Path("mqar_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    mqar = MQARConfig(
        vocab_size=args.vocab_size,
        num_kv_pairs=args.kv_pairs,
        num_queries=args.queries,
    )
    print(f"MQAR: {args.kv_pairs} kv / {args.queries} queries, "
          f"seq_len={2*(args.kv_pairs+args.queries)}", flush=True)

    results: dict = {"args": vars(args), "variants": {}}

    for variant in args.variants:
        print(f"\n=== variant: {variant} ===", flush=True)
        torch.manual_seed(args.seed)
        model = build_model(variant, args, vocab=args.vocab_size).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        summary = model.cfg.summary() if hasattr(model, "cfg") else f"DenseTransformer d={args.d_model} L={args.n_layers}"
        print(f"  params: {n_params:,}  |  {summary}", flush=True)

        t0 = time.time()
        history = train_mqar(
            model, mqar,
            steps=args.steps, batch_size=args.batch_size,
            lr=args.lr, device=device,
            eval_every=args.eval_every,
            on_log=lambda e: print(
                f"  step {e['step']:4d}  loss {e['train_loss']:.4f}  "
                f"eval_acc {e['eval_accuracy']*100:6.2f}%  lr {e['lr']:.2e}",
                flush=True,
            ),
            seed=args.seed,
        )
        dt = time.time() - t0

        final = history[-1]
        print(f"  final: acc {final['eval_accuracy']*100:.2f}%   ({dt:.1f}s)",
              flush=True)

        results["variants"][variant] = {
            "params": n_params,
            "time_s": dt,
            "history": history,
            "final_accuracy": final["eval_accuracy"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults -> {args.out}")

    print("\n=== SUMMARY ===")
    for v, r in results["variants"].items():
        print(f"  {v:10s}  acc = {r['final_accuracy']*100:6.2f}%   "
              f"params = {r['params']:,}")


if __name__ == "__main__":
    main()
