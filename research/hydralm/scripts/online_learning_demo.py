"""
Demonstration: the delta-rule state is a zero-gradient online learner.

Prints, in order:

1. A capacity curve -- recall accuracy vs N / head_dim -- showing the
   sharp phase transition at N ~ head_dim predicted by Widrow-Hoff LMS.
2. A headline memorization result at paper scale (1,000 facts, d=1024,
   H=4) showing 100% top-1 retrieval in a ~16 MB state, no gradients.
3. A destructive overwrite demo: writing (k, v_new) after (k, v_old)
   replaces v_old -- impossible with a raw KV cache.
4. The KV-cache accounting: how much memory a transformer would need
   to remember the same facts by keeping them in context.

Run:
    python scripts/online_learning_demo.py --scale paper   # shock numbers
    python scripts/online_learning_demo.py --scale smoke   # seconds
"""
from __future__ import annotations

import argparse
import time

import torch

from hydralm.eval.online_learning import (
    evaluate_capacity_curve,
    evaluate_memorization,
    evaluate_overwrite,
    kv_cache_memory_comparison,
)
from hydralm.memory import FactBank


SCALES = {
    "smoke": dict(d_head=128, n_heads=2, n_facts=64, kv_n_facts=1_000),
    "paper": dict(d_head=1024, n_heads=4, n_facts=1000, kv_n_facts=10_000),
}


def _ascii_bar(frac: float, width: int = 30) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "#" * n + "." * (width - n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=tuple(SCALES), default="smoke")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = SCALES[args.scale]
    d, H, N = cfg["d_head"], cfg["n_heads"], cfg["n_facts"]

    print("=" * 72)
    print(f" HydraLM FactBank -- zero-gradient test-time learning demo")
    print(f" scale = {args.scale}   d = {d}   heads = {H}   target N = {N}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Capacity curve on a single head (clean LMS theory).
    # ------------------------------------------------------------------
    print("\n[1/4] CAPACITY CURVE (single head -- cleanest view of LMS)")
    print(" fraction  N     cosine  argmax-top1")
    curve = evaluate_capacity_curve(
        head_dim=min(d, 256), n_heads=1, seed=args.seed,
    )
    for pt in curve["points"]:
        print(
            f"  {pt['fraction_of_d']:>5.3f}  "
            f"{pt['n_facts']:>4}  "
            f"{pt['cosine']:>6.3f}  "
            f"{pt['argmax_accuracy'] * 100:>5.1f}%  "
            f"|{_ascii_bar(pt['argmax_accuracy'])}|"
        )
    print("  ^-- phase transition around N = head_dim (Widrow 1960).")

    # ------------------------------------------------------------------
    # 2. Headline memorization.
    # ------------------------------------------------------------------
    print(f"\n[2/4] MEMORIZATION ({N} random (key, value) facts, d={d}, H={H})")
    t0 = time.perf_counter()
    mem = evaluate_memorization(
        n_facts=N, head_dim=d, n_heads=H, seed=args.seed,
    )
    dt = time.perf_counter() - t0
    mb = mem["state_bytes"] / (1024 ** 2)
    bpf = mem["bytes_per_fact"]
    print(f"  top-1 argmax accuracy : {mem['argmax_accuracy'] * 100:>6.2f}%")
    print(f"  mean cosine recall    : {mem['cosine']:>6.3f}  "
          f"(min across facts: {mem['cosine_min']:.3f})")
    print(f"  state memory          : {mb:>6.2f} MB "
          f"({bpf:,.0f} bytes per stored fact)")
    print(f"  writes required grad  : {mem['writes_require_grad']}")
    print(f"  time to write + query : {dt:.2f}s on CPU")

    # ------------------------------------------------------------------
    # 3. Destructive overwrite.
    # ------------------------------------------------------------------
    print("\n[3/4] DESTRUCTIVE OVERWRITE")
    print("      write (k, v_old)  ->  write (k, v_new)  ->  recall(k)")
    ow = evaluate_overwrite(n_facts=8, head_dim=64, n_heads=1, seed=args.seed)
    print(f"  cosine(recall, v_new) : {ow['cos_to_new']:>6.3f}")
    print(f"  cosine(recall, v_old) : {ow['cos_to_old']:>6.3f}")
    print(f"  overwrite margin      : {ow['overwrite_margin']:>6.3f}")
    print(f"  top-1 says 'v_new'    : {ow['argmax_to_new'] * 100:>6.1f}%")
    print("  ^-- a KV cache can only APPEND this pair; it cannot forget v_old.")

    # ------------------------------------------------------------------
    # 4. KV-cache accounting.
    # ------------------------------------------------------------------
    print(f"\n[4/4] MEMORY COMPARISON vs. equivalent transformer KV cache")
    Nkv = cfg["kv_n_facts"]
    kv = kv_cache_memory_comparison(
        n_facts=Nkv, head_dim=d, n_heads=H,
        n_layers=12, fact_token_length=16,
    )
    fb_mb = kv["factbank_bytes"] / (1024 ** 2)
    kv_mb = kv["transformer_kv_bytes"] / (1024 ** 2)
    print(f"  N = {Nkv:,} facts @ 16 tokens / fact, 12-layer Transformer")
    print(f"  FactBank state       : {fb_mb:>9.2f} MB  (O(H * d^2), fixed)")
    print(f"  Transformer KV cache : {kv_mb:>9.2f} MB  (O(N * layers * heads * d))")
    print(f"  RATIO                : {kv['ratio_transformer_over_factbank']:>9,.0f}x")

    # ------------------------------------------------------------------
    # Final message.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" TAKEAWAY")
    print("-" * 72)
    print(" The recurrent state in a gated DeltaNet layer is mathematically")
    print(" identical to one step of Widrow-Hoff LMS on  L(S) = ||S k - v||^2.")
    print(" That means every linear-attention model already contains an")
    print(" online least-squares solver. With a tiny API (`memorize` /")
    print(" `recall`) you can use it as an in-context memory store:")
    print("   - no fine-tuning")
    print("   - no optimiser")
    print("   - no gradient computation")
    print("   - state size is O(H * d^2), independent of N written")
    print("   - destructive overwrite (unlike a KV cache or RAG index)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
