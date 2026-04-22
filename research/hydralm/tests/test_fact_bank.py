"""
Unit tests for FactBank -- the mathematical guts behind Claim 6.

We deliberately test the MATH here (orthogonal-key perfect recall,
destructive overwrite, constant memory, no-grad writes) rather than
end-to-end retrieval accuracy; that end-to-end gate lives in
``tests/test_claims.py::test_claim_6_online_learning``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hydralm.memory import FactBank


def test_orthogonal_keys_give_exact_recall():
    """With one-hot keys, a single sweep of LMS stores each pair
    perfectly; recall(k_i) must equal v_i up to float32 noise."""
    d = 32
    torch.manual_seed(0)
    # Use head_init="identity" to isolate the math from the
    # multi-head rotation machinery.
    bank = FactBank(head_dim=d, n_heads=1, head_init="identity",
                    normalize_keys=False, seed=0)

    keys = torch.eye(d)                                  # orthogonal unit vectors
    values = torch.randn(d, d)
    bank.memorize(keys, values)
    out = bank.recall(keys, head_reduce="first")

    assert torch.allclose(out, values, atol=1e-4), \
        f"exact-recall residual too large: {(out - values).abs().max().item()}"


def test_memorize_does_not_require_grad():
    """Memorisation must be a pure tensor op. If anyone accidentally
    wraps it in an autograd graph, the resulting S would be huge and
    the 'zero-gradient' claim would be a lie."""
    bank = FactBank(head_dim=16, n_heads=2, seed=0)
    keys = torch.randn(4, 16, requires_grad=True)
    values = torch.randn(4, 16, requires_grad=True)
    stats = bank.memorize(keys, values)
    assert not bank.S.requires_grad
    # We still want the stats to REPORT if grad was requested on the
    # input, so callers can spot misuse:
    assert stats.writes_require_grad is True


def test_destructive_overwrite():
    """Writing (k, v_new) after (k, v_old) must leave S such that
    recall(k) is closer to v_new than to v_old. This is the property
    KV caches and RAG databases do not provide without extra logic."""
    d = 16
    bank = FactBank(head_dim=d, n_heads=1, head_init="identity",
                    normalize_keys=True, seed=0)
    k = F.normalize(torch.randn(1, d), dim=-1)
    v_old = torch.randn(1, d)
    v_new = torch.randn(1, d)

    bank.memorize(k, v_old)
    bank.memorize(k, v_new)
    out = bank.recall(k, head_reduce="first")

    cos_new = F.cosine_similarity(out, v_new).item()
    cos_old = F.cosine_similarity(out, v_old).item()
    assert cos_new > 0.99, f"recall did not snap to v_new: cos={cos_new}"
    assert cos_new > cos_old + 0.5, \
        f"overwrite margin too small: new={cos_new}, old={cos_old}"


def test_memory_is_constant_wrt_n():
    """state_bytes is a pure function of (n_heads, head_dim) -- writing
    more facts cannot possibly grow the state. This is the mechanical
    counterpart of Claim 3."""
    bank = FactBank(head_dim=32, n_heads=2, seed=0)
    before = bank.state_bytes
    keys = torch.randn(10, 32)
    vals = torch.randn(10, 32)
    bank.memorize(keys, vals)
    mid = bank.state_bytes
    for _ in range(50):
        keys = torch.randn(100, 32)
        vals = torch.randn(100, 32)
        bank.memorize(keys, vals)
    after = bank.state_bytes
    assert before == mid == after
    assert bank.num_facts_written == 10 + 50 * 100


def test_reset_wipes_state():
    bank = FactBank(head_dim=8, n_heads=1, seed=0)
    bank.memorize(torch.randn(3, 8), torch.randn(3, 8))
    assert bank.num_facts_written == 3
    bank.reset()
    assert bank.num_facts_written == 0
    assert torch.count_nonzero(bank.S).item() == 0


def test_random_key_recall_hits_argmax():
    """Even with random (non-orthogonal) keys, argmax retrieval must
    be >= 95% when N is comfortably below capacity (N <= d/4)."""
    torch.manual_seed(0)
    d = 128
    N = d // 4
    bank = FactBank(head_dim=d, n_heads=4, seed=0)
    keys = F.normalize(torch.randn(N, d), dim=-1)
    vals = torch.randn(N, d)
    bank.memorize(keys, vals)
    acc = bank.retrieval_accuracy(keys, vals)
    assert acc["argmax_accuracy"] >= 0.95, \
        f"argmax accuracy too low: {acc['argmax_accuracy']} (cosine={acc['cosine']})"


def test_head_averaging_reduces_noise():
    """More heads should not hurt and typically help: the mean-recall
    cosine across heads should be at least as high as a single head's,
    up to numerical noise."""
    torch.manual_seed(0)
    d, N = 128, 64
    keys = F.normalize(torch.randn(N, d), dim=-1)
    vals = torch.randn(N, d)

    bank1 = FactBank(head_dim=d, n_heads=1, seed=0)
    bank1.memorize(keys, vals)
    acc1 = bank1.retrieval_accuracy(keys, vals)

    bank8 = FactBank(head_dim=d, n_heads=8, seed=0)
    bank8.memorize(keys, vals)
    acc8 = bank8.retrieval_accuracy(keys, vals)

    # Small slack for float32 noise.
    assert acc8["cosine"] >= acc1["cosine"] - 1e-4, \
        f"8-head recall worse than 1-head: {acc8['cosine']} vs {acc1['cosine']}"
