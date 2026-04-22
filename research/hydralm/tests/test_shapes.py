"""Smoke tests: shapes, masking correctness, generation runs."""
from __future__ import annotations

import torch

from hydralm import HydraConfig, HydraLM, generate


def test_forward_shapes():
    cfg = HydraConfig(vocab_size=257, d_model=64, n_layers=4, n_heads=4,
                      swa_window=16, dn_chunk_size=8)
    model = HydraLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    out = model(x)
    assert out["logits"].shape == (2, 40, cfg.vocab_size)


def test_step_matches_forward_prefix():
    """Generating a sequence token-by-token must match the parallel forward."""
    torch.manual_seed(0)
    cfg = HydraConfig(vocab_size=257, d_model=64, n_layers=4, n_heads=4,
                      swa_window=16, dn_chunk_size=8,
                      dn_use_gate=False)   # silu*x nonlinearity adds fp noise; drop for tight bound
    model = HydraLM(cfg).double().eval()

    x = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        full = model(x)["logits"]

        state = None
        step_logits = []
        for t in range(32):
            lt, state = model.step(x[:, t], state)
            step_logits.append(lt)
    step_logits = torch.stack(step_logits, dim=1)

    diff = (full - step_logits).abs().max().item()
    assert diff < 1e-6, f"prefill/step divergence: {diff}"


def test_generate_runs():
    cfg = HydraConfig(vocab_size=257, d_model=64, n_layers=4, n_heads=4,
                      swa_window=16, dn_chunk_size=8)
    model = HydraLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 10))
    out = generate(model, prompt, max_new_tokens=20, temperature=0.0)
    assert out.shape == (2, 30)
