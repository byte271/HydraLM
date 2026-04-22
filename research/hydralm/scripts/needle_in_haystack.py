"""
Synthetic long-context recall task for ablation.

A "needle" key->value pair is inserted at a random position in a
sequence of random tokens, and the model must predict the value when
queried at the end:

    [  k1 v1   (filler ... filler)   k1  ?  ]
                                          ^-- should predict v1

This is the simplest possible probe of in-context associative recall,
and it is precisely the regime where linear-attention models
HISTORICALLY FAIL (Katharopoulos 2020, Schlag 2021) and where the
delta rule specifically was introduced to fix (Schlag 2021, Yang 2024).

We train a tiny HydraLM variant on this task and compute accuracy as a
function of haystack length.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from hydralm import HydraConfig, HydraLM
from hydralm.utils import seed_everything


def make_batch(B: int, N: int, V: int, device, gen: torch.Generator):
    """Return (input_ids, target_ids_at_last_pos)."""
    # reserve V-2, V-1 as special "key" and "query" markers
    KEY, QRY = V - 2, V - 1
    filler = torch.randint(0, V - 2, (B, N), generator=gen, device=device)
    pos = torch.randint(1, N - 4, (B,), generator=gen, device=device)  # where to place the (key, value) pair
    value = torch.randint(0, V - 2, (B,), generator=gen, device=device)

    x = filler.clone()
    b_idx = torch.arange(B, device=device)
    x[b_idx, pos] = KEY
    x[b_idx, pos + 1] = value
    # Query at the last two positions: KEY, then model must emit value.
    x[:, -2] = KEY
    x[:, -1] = value        # will be masked out from input; we predict it
    target = value
    return x, target


def train(args):
    seed_everything(0)
    device = torch.device(args.device)
    V = 128
    cfg = HydraConfig(
        vocab_size=V, d_model=128, n_layers=4, n_heads=4, head_dim=32,
        swa_every=99,                     # pure DeltaNet, no SWA, for ablation
        dn_chunk_size=32,
        max_position_embeddings=max(args.lengths),
    )
    model = HydraLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(0)

    model.train()
    print(f"Training tiny HydraLM ({model.num_parameters():,} params) on synthetic recall...")
    for step in range(args.steps):
        N = args.lengths[step % len(args.lengths)]
        x, tgt = make_batch(args.batch_size, N, V, device, gen)
        logits = model(x)["logits"][:, -2, :]      # predict from the query KEY position
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            acc = (logits.argmax(-1) == tgt).float().mean().item()
            print(f"  step {step:4d}  N={N:5d}  loss={loss.item():.4f}  acc={acc:.3f}")

    # Eval across lengths
    model.eval()
    print("\nLength   Accuracy")
    with torch.no_grad():
        for N in args.lengths:
            correct = total = 0
            for _ in range(args.eval_batches):
                x, tgt = make_batch(args.batch_size, N, V, device, gen)
                logits = model(x)["logits"][:, -2, :]
                correct += (logits.argmax(-1) == tgt).sum().item()
                total += tgt.numel()
            print(f"  {N:5d}    {correct / total:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    train(ap.parse_args())
