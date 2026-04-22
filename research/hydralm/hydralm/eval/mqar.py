"""
Multi-Query Associative Recall (MQAR) — the Zoology benchmark.

Reference:
    Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language
    Models", ICLR 2024. https://arxiv.org/abs/2312.04927

Task setup
----------
Each sequence is a list of key/value pairs followed by a list of query keys.
The model must predict, for each query position, the value that was paired with
the matching key earlier in the sequence. This is the single cleanest probe of
in-context associative recall, which is where linear-attention models
historically fail and where Gated DeltaNet (and hybrid SWA) are designed to
recover.

The data generator here follows the exact protocol used in the Zoology paper:

    k_1 v_1 k_2 v_2 ... k_D v_D q_1 a_1 q_2 a_2 ... q_Q a_Q

where `k_i v_i` are randomly-drawn key/value pairs, `q_j` is a key sampled from
the D pairs, and `a_j` is the corresponding value. Loss / accuracy is computed
ONLY on the `a_j` positions (the rest are ignored via `ignore_index = -100`).

This file exposes:
  - ``make_mqar_batch``: pure-PyTorch data generator, no external deps.
  - ``evaluate_mqar``: runs a forward pass and returns recall accuracy.
  - ``train_mqar``: small trainer that fits a model on MQAR and reports the
    learning curve (for ablations and architecture comparison).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Ignore-index for CE loss on non-answer positions.
IGNORE = -100


@dataclass
class MQARConfig:
    """Specification of an MQAR instance distribution."""
    vocab_size: int = 8192
    num_kv_pairs: int = 64
    num_queries: int = 64
    seq_len: int | None = None  # defaults to 2*(num_kv_pairs + num_queries)


def make_mqar_batch(
    batch_size: int,
    cfg: MQARConfig,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate a batch of MQAR sequences.

    Returns
    -------
    input_ids : LongTensor, shape (B, N)
    labels    : LongTensor, shape (B, N). ``IGNORE`` everywhere except at the
                positions immediately following a query token, where it holds
                the true answer value id.

    Protocol
    --------
    Keys are drawn from the lower half of the vocabulary [0, V/2), values from
    the upper half [V/2, V). This guarantees keys and values never collide and
    that the model cannot "cheat" by confusing a query with its own answer slot.
    """
    V = cfg.vocab_size
    D = cfg.num_kv_pairs
    Q = cfg.num_queries
    N = cfg.seq_len or 2 * (D + Q)
    assert N >= 2 * (D + Q), f"seq_len {N} too short for {D} kv pairs + {Q} queries"

    g = generator
    key_space = V // 2
    val_space = V - key_space
    val_offset = key_space

    ids = torch.zeros(batch_size, N, dtype=torch.long, device=device)
    lbl = torch.full((batch_size, N), IGNORE, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Sample D *distinct* keys so the mapping is a well-defined function.
        keys = torch.randperm(key_space, generator=g, device=device)[:D]
        values = torch.randint(
            low=val_offset, high=V, size=(D,), generator=g, device=device
        )
        kv_map = dict(zip(keys.tolist(), values.tolist()))

        # Layout: kv pairs interleaved, then queries interleaved with answers.
        pos = 0
        for k, v in zip(keys.tolist(), values.tolist()):
            ids[b, pos] = k
            ids[b, pos + 1] = v
            pos += 2

        # Queries: sample from the stored keys WITH replacement (the model
        # must handle repeats, which is the hard case).
        query_idx = torch.randint(0, D, (Q,), generator=g, device=device)
        for j in range(Q):
            k = keys[query_idx[j]].item()
            a = kv_map[k]
            ids[b, pos] = k
            ids[b, pos + 1] = a
            # The model sees the query at position `pos` and must predict `a`
            # at position `pos + 1`, so the *label at position `pos`* is `a`
            # (next-token prediction semantics).
            lbl[b, pos] = a
            pos += 2

    return ids, lbl


@torch.no_grad()
def evaluate_mqar(
    model: nn.Module,
    cfg: MQARConfig,
    n_batches: int = 32,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Compute MQAR recall accuracy + loss on freshly-sampled batches."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    correct, total, loss_sum, loss_count = 0, 0, 0.0, 0

    for _ in range(n_batches):
        ids, lbl = make_mqar_batch(batch_size, cfg, device=device, generator=g)
        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out
        # Next-token shift: predict position t+1 from position t.
        logits = logits[:, :-1]
        targets = lbl[:, :-1]
        mask = targets != IGNORE
        if mask.any():
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE,
                reduction="sum",
            )
            loss_sum += loss.item()
            loss_count += int(mask.sum().item())
            pred = logits.argmax(dim=-1)
            correct += int(((pred == targets) & mask).sum().item())
            total += int(mask.sum().item())

    acc = correct / max(total, 1)
    loss = loss_sum / max(loss_count, 1)
    return {"mqar_accuracy": acc, "mqar_loss": loss, "answer_positions": total}


def train_mqar(
    model: nn.Module,
    cfg: MQARConfig,
    steps: int = 2000,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str | torch.device = "cpu",
    eval_every: int = 200,
    on_log: Callable[[dict], None] | None = None,
    seed: int = 0,
    warmup_ratio: float = 0.1,
) -> list[dict]:
    """Fit ``model`` on freshly sampled MQAR batches. Returns a learning-curve
    list of dicts with keys ``step``, ``train_loss``, ``eval_accuracy``.

    Parameters
    ----------
    warmup_ratio : float
        Fraction of total steps for linear LR warmup (standard LM practice).
        The remaining steps follow cosine decay to zero. Set to ``0.0`` to
        reproduce pre-warmup behaviour.
    """
    # Decoupled weight decay: norms, biases, and embeddings don't get decay.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "embed" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(steps * warmup_ratio))

    def _lr_mul(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mul)
    g = torch.Generator(device=device).manual_seed(seed)

    history: list[dict] = []
    model.train()

    for step in range(1, steps + 1):
        ids, lbl = make_mqar_batch(batch_size, cfg, device=device, generator=g)
        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out
        logits = logits[:, :-1]
        targets = lbl[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            metrics = evaluate_mqar(
                model, cfg, n_batches=8, batch_size=batch_size,
                device=device, seed=seed + 9999,
            )
            entry = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "eval_accuracy": metrics["mqar_accuracy"],
                "eval_loss": metrics["mqar_loss"],
                "lr": sched.get_last_lr()[0],
            }
            history.append(entry)
            if on_log is not None:
                on_log(entry)
            model.train()

    return history


__all__ = ["MQARConfig", "make_mqar_batch", "evaluate_mqar", "train_mqar", "IGNORE"]
