"""
RULER-style synthetic long-context retrieval probes.

Reference:
    Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context
    Language Models?", 2024. https://arxiv.org/abs/2404.06654

We implement two of the most diagnostic RULER subtasks, adapted to a
synthetic vocabulary so the evaluation is self-contained (no tokenizer
dependencies):

  1. ``single_nih`` — Single Needle-in-a-Haystack: one key/value pair is
     inserted at a random depth in a long sequence of filler tokens; the
     final query asks for the value.

  2. ``multi_nih``  — Multi-NIH: K key/value pairs at different depths.
     The query at the end asks for one of them at random. This is a
     strictly harder retrieval task because the model must *discriminate*
     between co-present keys, not merely recall the last-seen one.

Both tasks share a common protocol: the model's loss / accuracy is only
measured at the final answer position, so filler tokens are irrelevant
training signal.

Needle layout
-------------
Each needle occupies exactly 4 token slots:

    [KEY_MARK, key_token, VAL_MARK, value_token]

The K needle start positions are drawn from a uniform random permutation of
*aligned* slots separated by at least 4 positions so that no two needles
ever overlap.  The guarantee is maintained by sampling from
``[0, 4, 8, ..., max_needle_pos]`` (step 4) rather than any arbitrary offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


IGNORE = -100


@dataclass
class LongContextConfig:
    vocab_size: int = 8192
    seq_len: int = 8192              # total sequence length
    num_needles: int = 1             # 1 = single_nih, >1 = multi_nih
    # Filler is sampled from the "low" token range; keys/values from the
    # high range, so the filler cannot accidentally match a key.
    filler_frac: float = 0.5


def make_needle_batch(
    batch_size: int,
    cfg: LongContextConfig,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, list[int]]:
    """Generate a batch of Needle-in-a-Haystack sequences.

    Returns
    -------
    ids    : (B, N) int64 token ids.
    labels : (B, N) int64, ``IGNORE`` everywhere except the final-answer
             position (the value token in the query trailer), where it holds
             the true answer.
    depths : list of length B giving the absolute position of the queried
             needle — useful for computing a "recall at depth" curve.

    Layout
    ------
    Each needle is stored as four consecutive tokens::

        [KEY_MARK, key_tok, VAL_MARK, value_tok]

    Needle start positions are drawn from multiples of 4 so that two needles
    never share any of their four token slots (a bug present in some earlier
    implementations that made multi-NIH trivially easier than intended).

    The sequence ends with a fixed query trailer (4 tokens)::

        [QUERY_MARK, query_key_tok, VAL_MARK, answer_tok]

    Loss is measured at the position that predicts ``answer_tok``.
    """
    V = cfg.vocab_size
    N = cfg.seq_len
    K = cfg.num_needles
    filler_hi = int(V * cfg.filler_frac)
    kv_lo = filler_hi

    # Reserve three special marker tokens at the top of the vocabulary.
    KEY_MARK = V - 1
    VAL_MARK = V - 2
    QUERY_MARK = V - 3
    kv_hi = V - 3   # exclusive upper bound for content key/value tokens
    assert kv_hi > kv_lo + 4, (
        f"vocab_size={V} too small: kv_hi={kv_hi} must be > kv_lo+4={kv_lo + 4}"
    )

    # Query trailer occupies the last 4 token positions.
    trailer_len = 4

    # Each needle occupies 4 slots; needles are placed at multiples-of-4
    # positions so they never overlap.  We must leave room for the trailer.
    max_aligned_start = (N - trailer_len) // 4 - K
    assert max_aligned_start >= K, (
        f"seq_len={N} too short to place {K} non-overlapping needles "
        f"with a {trailer_len}-token query trailer."
    )
    # Number of valid aligned positions for the *first* needle.
    n_slots = max_aligned_start + 1  # positions 0, 4, 8, ..., max_aligned_start*4

    ids = torch.randint(0, filler_hi, (batch_size, N), generator=generator, device=device)
    lbl = torch.full((batch_size, N), IGNORE, dtype=torch.long, device=device)
    depths: list[int] = []

    for b in range(batch_size):
        # Sample K distinct aligned slots without replacement.
        slot_indices = torch.randperm(n_slots, generator=generator, device=device)[:K]
        # Convert slot indices to actual byte positions (multiples of 4).
        positions = slot_indices * 4

        keys = torch.randint(kv_lo, kv_hi, (K,), generator=generator, device=device)
        values = torch.randint(kv_lo, kv_hi, (K,), generator=generator, device=device)

        for p, k_tok, v_tok in zip(
            positions.tolist(), keys.tolist(), values.tolist()
        ):
            ids[b, p]     = KEY_MARK
            ids[b, p + 1] = k_tok
            ids[b, p + 2] = VAL_MARK
            ids[b, p + 3] = v_tok

        # Choose which needle to query.
        pick = int(
            torch.randint(0, K, (1,), generator=generator, device=device).item()
        )
        q_pos = N - trailer_len
        ids[b, q_pos]     = QUERY_MARK
        ids[b, q_pos + 1] = int(keys[pick].item())
        ids[b, q_pos + 2] = VAL_MARK
        ids[b, q_pos + 3] = int(values[pick].item())

        # Next-token prediction: predicting position (q_pos+3) from position
        # (q_pos+2), so the label is at position q_pos+2.
        lbl[b, q_pos + 2] = int(values[pick].item())

        depths.append(int(positions[pick].item()))

    return ids, lbl, depths


@torch.no_grad()
def evaluate_needle(
    model: nn.Module,
    cfg: LongContextConfig,
    n_batches: int = 8,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Compute recall accuracy at the query position and a depth-bucket
    breakdown (4 equal-width depth buckets covering [0, 1)).

    Returns a dict with keys:
        ``accuracy``, ``acc_depth_0_25``, ``acc_depth_25_50``,
        ``acc_depth_50_75``, ``acc_depth_75_100``.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    bucket_correct = [0, 0, 0, 0]
    bucket_total   = [0, 0, 0, 0]
    total_correct  = 0
    total          = 0

    for _ in range(n_batches):
        ids, lbl, depths = make_needle_batch(
            batch_size, cfg, device=device, generator=g
        )
        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out

        # The label for sequence b is at position q_pos+2; the model
        # predicts it from position q_pos+2 (i.e. logits at q_pos+2-1).
        target_lbl_pos = cfg.seq_len - 2     # q_pos + 2
        pred = logits[:, target_lbl_pos - 1].argmax(dim=-1)   # (B,)
        target = lbl[:, target_lbl_pos]                        # (B,)
        correct = pred == target                               # (B,) bool

        for i, d in enumerate(depths):
            frac = d / cfg.seq_len
            bucket = min(int(frac / 0.25), 3)   # 0..3
            bucket_total[bucket]   += 1
            bucket_correct[bucket] += int(correct[i].item())

        total_correct += int(correct.sum().item())
        total         += correct.numel()

    return {
        "accuracy":        total_correct / max(total, 1),
        "acc_depth_0_25":  bucket_correct[0] / max(bucket_total[0], 1),
        "acc_depth_25_50": bucket_correct[1] / max(bucket_total[1], 1),
        "acc_depth_50_75": bucket_correct[2] / max(bucket_total[2], 1),
        "acc_depth_75_100": bucket_correct[3] / max(bucket_total[3], 1),
    }


__all__ = ["LongContextConfig", "make_needle_batch", "evaluate_needle"]
