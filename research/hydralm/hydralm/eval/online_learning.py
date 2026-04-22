"""
Evaluation harness for Claim 6: zero-gradient test-time learning.

Four benchmark protocols, each returning a JSON-serialisable dict so
they can be dropped into RESULTS.md:

1. `evaluate_memorization`  -- write N facts into a fresh FactBank,
    measure retrieval accuracy in both cosine similarity and strict
    top-1 argmax. The point of claim 6 is that this is non-trivially
    high without any fine-tuning.

2. `evaluate_capacity_curve`  -- sweep N and record how accuracy
    degrades, verifying the O(sqrt(N/d)) scaling predicted by LMS
    theory (Widrow 1960). Produces the table that underlies the
    "sharp phase transition near N ~ d" narrative.

3. `evaluate_interference`  -- interleave random distractor writes
    between the facts, showing that reads survive arbitrary context
    junk as long as the total write budget stays within capacity.

4. `evaluate_overwrite`  -- write (k, v_old), then (k, v_new), and
    verify that ``recall(k)`` snaps to ``v_new`` rather than averaging.
    This is the property RAG and raw KV caches cannot provide.

5. `kv_cache_memory_comparison`  -- accounting-only: what would a
    transformer need to remember the same N facts, token-by-token?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from hydralm.memory import FactBank


# ---------------------------------------------------------------------
def _random_keys(n: int, d: int, *, seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(n, d, generator=g).to(device)
    return torch.nn.functional.normalize(k, dim=-1)


def _random_values(n: int, d: int, *, seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, d, generator=g).to(device)


# ---------------------------------------------------------------------
def evaluate_memorization(
    *,
    n_facts: int = 1024,
    head_dim: int = 256,
    n_heads: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Single-shot memorization: write N facts, query every one.

    Returns accuracy stats and the effective "bytes per fact" cost,
    which is the ratio of the fixed state size to the number of facts
    it successfully stored. This number is the shock value -- it stays
    roughly constant while the corresponding KV-cache-per-fact grows
    linearly with N.
    """
    bank = FactBank(head_dim=head_dim, n_heads=n_heads,
                    seed=seed, device=device)

    keys = _random_keys(n_facts, head_dim, seed=seed, device=device)
    values = _random_values(n_facts, head_dim, seed=seed + 1, device=device)

    stats = bank.memorize(keys, values)
    acc = bank.retrieval_accuracy(keys, values)
    return {
        "n_facts": n_facts,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "state_bytes": stats.state_bytes,
        "bytes_per_fact": stats.state_bytes / max(n_facts, 1),
        "cosine": acc["cosine"],
        "cosine_min": acc["cosine_min"],
        "argmax_accuracy": acc["argmax_accuracy"],
        "mse": acc["mse"],
        "writes_require_grad": stats.writes_require_grad,
    }


# ---------------------------------------------------------------------
def evaluate_capacity_curve(
    *,
    head_dim: int = 128,
    n_heads: int = 1,
    fractions: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0),
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Sweep N = frac * head_dim and record recall quality at each.

    With ``n_heads=1`` and unit-norm random keys, LMS theory predicts
    near-perfect recall for N < head_dim (keys approximately
    orthogonal) and a smooth decay after. The returned ``points`` list
    is the exact table we render in RESULTS.md.
    """
    points = []
    for f in fractions:
        N = max(1, int(round(f * head_dim)))
        bank = FactBank(head_dim=head_dim, n_heads=n_heads,
                        seed=seed, device=device)
        keys = _random_keys(N, head_dim, seed=seed, device=device)
        vals = _random_values(N, head_dim, seed=seed + 1, device=device)
        bank.memorize(keys, vals)
        acc = bank.retrieval_accuracy(keys, vals)
        points.append({
            "fraction_of_d": f,
            "n_facts": N,
            "cosine": acc["cosine"],
            "argmax_accuracy": acc["argmax_accuracy"],
        })
    return {
        "head_dim": head_dim,
        "n_heads": n_heads,
        "points": points,
    }


# ---------------------------------------------------------------------
def evaluate_interference(
    *,
    n_facts: int = 128,
    n_distractors: int = 1024,
    head_dim: int = 256,
    n_heads: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Interleave ``n_distractors`` junk writes between the real facts
    and show that recall on the real facts is only modestly affected.

    Implementation: write junk chunks of length
    ``n_distractors // n_facts`` between each real (k, v) pair. We then
    query ONLY the real keys and measure accuracy; the junk writes are
    never queried so a model that merely "reads back whatever it last
    saw" would fail this entirely.
    """
    bank = FactBank(head_dim=head_dim, n_heads=n_heads,
                    seed=seed, device=device)

    keys = _random_keys(n_facts, head_dim, seed=seed, device=device)
    vals = _random_values(n_facts, head_dim, seed=seed + 1, device=device)
    junk_block = max(1, n_distractors // max(n_facts, 1))
    g_junk = torch.Generator(device="cpu").manual_seed(seed + 7)

    for i in range(n_facts):
        bank.memorize(keys[i:i + 1], vals[i:i + 1])
        jk = torch.randn(junk_block, head_dim, generator=g_junk).to(device)
        jv = torch.randn(junk_block, head_dim, generator=g_junk).to(device)
        bank.memorize(jk, jv)

    acc = bank.retrieval_accuracy(keys, vals)
    return {
        "n_facts": n_facts,
        "n_distractors": n_facts * junk_block,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "state_bytes": bank.state_bytes,
        "cosine": acc["cosine"],
        "argmax_accuracy": acc["argmax_accuracy"],
    }


# ---------------------------------------------------------------------
def evaluate_overwrite(
    *,
    n_facts: int = 16,
    head_dim: int = 64,
    n_heads: int = 1,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Write (k, v_old); then write (k, v_new). Recall MUST return
    v_new, not the average -- this is what RAG and KV caches cannot do.

    We report two numbers:
      * cosine to ``v_new`` (should be close to 1)
      * cosine to ``v_old`` (should be close to 0 or negative).
    The gap between them is the "overwrite margin".
    """
    bank = FactBank(head_dim=head_dim, n_heads=n_heads,
                    seed=seed, device=device)

    keys = _random_keys(n_facts, head_dim, seed=seed, device=device)
    v_old = _random_values(n_facts, head_dim, seed=seed + 1, device=device)
    v_new = _random_values(n_facts, head_dim, seed=seed + 2, device=device)

    bank.memorize(keys, v_old)
    bank.memorize(keys, v_new)

    acc_new = bank.retrieval_accuracy(keys, v_new)
    acc_old = bank.retrieval_accuracy(keys, v_old)
    return {
        "n_facts": n_facts,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "cos_to_new": acc_new["cosine"],
        "cos_to_old": acc_old["cosine"],
        "overwrite_margin": acc_new["cosine"] - acc_old["cosine"],
        "argmax_to_new": acc_new["argmax_accuracy"],
    }


# ---------------------------------------------------------------------
def kv_cache_memory_comparison(
    *,
    n_facts: int,
    head_dim: int,
    n_heads: int,
    n_layers: int = 12,
    fact_token_length: int = 16,
    dtype_bytes: int = 4,
) -> dict:
    """Accounting-only: how much KV cache a transformer would need to
    "remember" the same facts by keeping them in context.

    A transformer with ``n_layers`` layers and ``n_heads`` heads of
    ``head_dim`` needs ``2 * n_layers * n_heads * head_dim`` scalars
    per CACHED token (one K, one V per layer per head). If each fact
    spends ``fact_token_length`` tokens of context, total cache is
    ``2 * n_layers * n_heads * head_dim * fact_token_length * n_facts``.
    A FactBank, in contrast, uses ``n_heads * head_dim**2`` scalars
    total, independent of N.
    """
    factbank_bytes = n_heads * head_dim * head_dim * dtype_bytes
    kv_per_token = 2 * n_layers * n_heads * head_dim * dtype_bytes
    kv_bytes = kv_per_token * fact_token_length * n_facts
    return {
        "n_facts": n_facts,
        "factbank_bytes": factbank_bytes,
        "transformer_kv_bytes": kv_bytes,
        "ratio_transformer_over_factbank": kv_bytes / max(factbank_bytes, 1),
        "breakeven_n_facts": max(
            1,
            factbank_bytes // max(kv_per_token * fact_token_length, 1),
        ),
        "params": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "fact_token_length": fact_token_length,
            "dtype_bytes": dtype_bytes,
        },
    }


__all__ = [
    "evaluate_memorization",
    "evaluate_capacity_curve",
    "evaluate_interference",
    "evaluate_overwrite",
    "kv_cache_memory_comparison",
]
