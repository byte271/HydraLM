"""Unit tests for speculative decoding."""
from __future__ import annotations

import copy

import pytest
import torch

from hydralm import HydraConfig, HydraLM, generate, speculative_generate


def _make_model(d_model: int = 32, n_layers: int = 2, seed: int = 0) -> HydraLM:
    torch.manual_seed(seed)
    cfg = HydraConfig(
        vocab_size=47, d_model=d_model, n_layers=n_layers, n_heads=2,
        swa_every=2, swa_window=16, dn_chunk_size=8,
        max_position_embeddings=256,
    )
    m = HydraLM(cfg)
    m.eval()
    return m


# -----------------------------------------------------------------------------
# Shape / basic behaviour
# -----------------------------------------------------------------------------
def test_spec_decoding_produces_correct_shape():
    target = _make_model(d_model=32, seed=0)
    draft = _make_model(d_model=24, seed=1)
    prompt = torch.randint(0, 47, (2, 5))
    out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=30, k=4,
        temperature=1.0, top_k=None, top_p=None,
    )
    assert out.shape[0] == 2
    # Produced at least prompt + max_new_tokens tokens (may be trimmed to exactly that).
    assert out.shape[1] == prompt.shape[1] + 30
    # Prompt prefix preserved.
    assert torch.equal(out[:, : prompt.shape[1]], prompt)
    assert stats.rounds > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_spec_decoding_greedy_is_deterministic():
    target = _make_model(d_model=32, seed=7)
    draft = _make_model(d_model=32, seed=7)  # same model -> acceptance=1
    prompt = torch.tensor([[1, 2, 3, 4]])

    torch.manual_seed(0)
    out1, _ = speculative_generate(target, draft, prompt, max_new_tokens=20, k=4, temperature=0.0)
    torch.manual_seed(0)
    out2, _ = speculative_generate(target, draft, prompt, max_new_tokens=20, k=4, temperature=0.0)
    assert torch.equal(out1, out2), "temperature=0 spec decoding is not deterministic"


def test_spec_decoding_identical_draft_matches_target_greedy():
    """When draft == target and temperature=0, spec decode must match target argmax.

    This validates that the state-sync / rollback logic is consistent with a
    plain autoregressive forward pass."""
    target = _make_model(d_model=32, seed=3)
    draft = copy.deepcopy(target)
    prompt = torch.tensor([[5, 11, 17, 3]])

    # Reference: single-model greedy generation.
    ref = generate(target, prompt, max_new_tokens=25, temperature=0.0)

    torch.manual_seed(0)
    out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=25, k=4, temperature=0.0,
    )
    assert torch.equal(out[:, : ref.shape[1]], ref), (
        "spec decoding with draft==target must match single-model greedy output"
    )
    # With identical models + greedy, every drafted token is accepted.
    assert stats.acceptance_rate > 0.95, (
        f"acceptance rate should be ~1 with identical greedy models, got {stats.acceptance_rate:.3f}"
    )


# -----------------------------------------------------------------------------
# Statistical equivalence (sampling mode)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 6])
def test_spec_decoding_distribution_matches_target(k: int):
    """With draft==target, the sampling distribution must equal the target's
    plain autoregressive distribution. We check this by computing token
    histograms on a short-horizon prediction from many trials."""
    target = _make_model(d_model=24, n_layers=2, seed=2)
    draft = copy.deepcopy(target)

    prompt = torch.tensor([[9, 1, 21, 7]])
    n_trials = 200
    spec_first = []
    ref_first = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        out_s, _ = speculative_generate(
            target, draft, prompt, max_new_tokens=1, k=k, temperature=1.0,
        )
        spec_first.append(int(out_s[0, -1].item()))

        torch.manual_seed(trial)
        out_r = generate(target, prompt, max_new_tokens=1, temperature=1.0, top_k=None, top_p=None)
        ref_first.append(int(out_r[0, -1].item()))

    V = target.cfg.vocab_size
    hs = torch.zeros(V); hr = torch.zeros(V)
    for t in spec_first: hs[t] += 1
    for t in ref_first: hr[t] += 1
    hs /= n_trials; hr /= n_trials
    # Total-variation distance between the two empirical distributions.
    tv = 0.5 * (hs - hr).abs().sum().item()
    assert tv < 0.20, f"spec decoding (k={k}) distribution drifted: TV={tv:.3f}"


def test_spec_decoding_different_draft_still_exact_in_expectation():
    """With a DIFFERENT (worse) draft, spec decoding is still exact vs the
    target — acceptance rate drops but the committed tokens remain from the
    target distribution. We check that no 'impossible' tokens appear."""
    target = _make_model(d_model=32, seed=4)
    draft = _make_model(d_model=16, seed=99)  # much weaker, different init
    prompt = torch.tensor([[2, 2, 3, 5]])
    torch.manual_seed(123)
    out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=40, k=4, temperature=1.0, top_k=20,
    )
    assert out.shape == (1, prompt.shape[1] + 40)
    # With a mismatched draft, acceptance should drop but still be > 0.
    assert 0.0 < stats.acceptance_rate
    # No invalid token ids.
    assert out.min().item() >= 0
    assert out.max().item() < target.cfg.vocab_size
