"""
Train a tiny HydraLM on a character-level corpus.

This is a minimal training loop that exercises every code path:
  * chunkwise DeltaNet kernel
  * SWA layer
  * SwiGLU MLP, RMSNorm, tied embeddings
  * gradient clipping, warmup, cosine decay

Run with any text file:
    python scripts/train_tiny.py --data path/to/text.txt --steps 2000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hydralm import HydraConfig, HydraLM
from hydralm.utils import count_parameters, seed_everything


def load_char_dataset(path: Path) -> tuple[torch.Tensor, dict[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, itos


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix]).to(device)
    return x, y


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="Optimiser backend. 'muon' uses hybrid Muon (2D matrices) + AdamW (embeds/head/bias/norm).")
    ap.add_argument("--muon-lr", type=float, default=5e-3)
    args = ap.parse_args()

    seed_everything(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    data, itos = load_char_dataset(args.data)
    vocab_size = len(itos)
    print(f"Loaded {len(data):,} tokens, vocab {vocab_size}")

    cfg = HydraConfig(
        vocab_size=vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads,
        swa_every=4, swa_window=128, dn_chunk_size=64,
        max_position_embeddings=max(args.seq_len * 8, 8192),
    )
    model = HydraLM(cfg).to(device=device, dtype=dtype)
    print(f"{cfg.summary()}  |  params = {count_parameters(model):,}")

    if args.optimizer == "muon":
        from hydralm.optim import build_hybrid_optimizer
        opt = build_hybrid_optimizer(
            model,
            muon_lr=args.muon_lr,
            muon_momentum=0.95,
            muon_weight_decay=0.0,
            adamw_lr=args.lr,
            adamw_betas=(0.9, 0.95),
            adamw_weight_decay=0.1,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Record the initial LR per param-group so Muon/AdamW keep their distinct base LRs.
    base_lrs = [g["lr"] for g in opt.param_groups]
    model.train()
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = cosine_warmup(step, args.warmup, args.steps, base)

        x, y = get_batch(data, args.batch_size, args.seq_len, device)
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % args.log_every == 0:
            recent = sum(losses[-args.log_every:]) / args.log_every
            ppl = math.exp(min(recent, 20))
            print(f"step {step:5d}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"loss {recent:.4f}  ppl {ppl:.1f}")

        if step % args.sample_every == 0:
            sample = do_sample(model, itos, device, n=160)
            print(f"\n--- sample @ step {step} ---\n{sample}\n----------------\n")


@torch.no_grad()
def do_sample(model, itos, device, n=200, temperature=0.8):
    from hydralm import generate
    vocab = len(itos)
    start = torch.zeros(1, 1, dtype=torch.long, device=device)
    out = generate(model, start, max_new_tokens=n, temperature=temperature, top_k=40)
    return "".join(itos.get(int(t), "?") for t in out[0].tolist())


if __name__ == "__main__":
    main()
