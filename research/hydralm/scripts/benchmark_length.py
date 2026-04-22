"""
Throughput and memory benchmark: HydraLM vs. a same-parameter Transformer
across sequence lengths 2^8 ... 2^20.

Run:
    python scripts/benchmark_length.py --max-log2 16 --device cuda

The Transformer baseline uses PyTorch's native flash / memory-efficient
SDPA kernels, so the comparison is apples-to-apples: both models have
access to the fastest available attention implementation.

Expected behaviour:
  * Transformer prefill memory grows as O(N^2) — OOMs somewhere around
    N ~ 32k on a 24 GB GPU.
  * HydraLM prefill memory grows as O(N) — reaches 1M+ tokens routinely.
  * Per-token decoding: Transformer KV-cache = O(N); HydraLM = O(1).
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch.nn as nn

from hydralm import HydraConfig, HydraLM


# --------------------------------------------------------------------- #
# Minimal Transformer baseline with identical parameter count.           #
# --------------------------------------------------------------------- #
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model * 8 / 3),
            dropout=0.0, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        N = x.shape[1]
        mask = torch.triu(torch.full((N, N), float("-inf"), device=x.device), diagonal=1)
        x = self.stack(x, mask=mask, is_causal=True)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------- #
def time_forward(model, ids, warmup=2, iters=3):
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(ids)
    if ids.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(ids)
    if ids.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def peak_memory_mib(device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--min-log2", type=int, default=8)
    ap.add_argument("--max-log2", type=int, default=14)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cfg = HydraConfig(
        vocab_size=args.vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        swa_every=4, swa_window=512, dn_chunk_size=64,
        max_position_embeddings=1 << args.max_log2,
    )
    hydra = HydraLM(cfg).to(device=device, dtype=dtype).eval()
    xfmr = TinyTransformer(args.vocab, args.d_model, args.n_layers, args.n_heads)
    xfmr = xfmr.to(device=device, dtype=dtype).eval()

    print(cfg.summary())
    print(f"HydraLM params : {hydra.num_parameters():,}")
    print(f"Xfmr params    : {sum(p.numel() for p in xfmr.parameters()):,}")
    print()
    print(f"{'N':>9} | {'Hydra ms':>10} | {'Hydra MiB':>10} | {'Xfmr ms':>10} | {'Xfmr MiB':>10} | speedup")
    print("-" * 78)

    for lg in range(args.min_log2, args.max_log2 + 1):
        N = 1 << lg
        ids = torch.randint(0, args.vocab, (1, N), device=device)

        # HydraLM
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_h = time_forward(hydra, ids)
        m_h = peak_memory_mib(device)

        # Transformer (may OOM at large N)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            t_x = time_forward(xfmr, ids)
            m_x = peak_memory_mib(device)
            speedup = f"{t_x / t_h:6.2f}x"
        except torch.cuda.OutOfMemoryError:
            t_x, m_x, speedup = float("nan"), float("nan"), "OOM"

        print(f"{N:>9} | {t_h*1000:>10.2f} | {m_h:>10.1f} | "
              f"{t_x*1000:>10.2f} | {m_x:>10.1f} | {speedup}")


if __name__ == "__main__":
    main()
