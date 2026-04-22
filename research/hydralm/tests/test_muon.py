"""Unit tests for the Muon optimizer."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hydralm import HydraConfig, HydraLM
from hydralm.optim import (
    HybridMuonAdamW,
    Muon,
    build_hybrid_optimizer,
    zeropower_via_newton_schulz,
)


# -----------------------------------------------------------------------------
# Newton-Schulz approximate orthogonalisation
# -----------------------------------------------------------------------------
def test_newton_schulz_maps_to_near_orthogonal():
    """After 5 NS steps, singular values should all land in roughly [0.65, 1.35].

    Keller's NS coefficients (3.4445, -4.7750, 2.0315) are tuned for bf16
    stability rather than exact orthogonalisation, so they converge to a
    bounded band around 1, not to exactly 1.
    """
    torch.manual_seed(0)
    G = torch.randn(64, 48)
    O = zeropower_via_newton_schulz(G, steps=5)
    s = torch.linalg.svdvals(O)
    assert s.min() > 0.65, f"min singular value too small: {s.min().item()}"
    assert s.max() < 1.35, f"max singular value too large: {s.max().item()}"


def test_newton_schulz_preserves_singular_directions():
    """NS should produce U @ V.T, i.e. singular VECTORS of G, not random ones."""
    torch.manual_seed(1)
    G = torch.randn(32, 32)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    expected = U @ Vh
    O = zeropower_via_newton_schulz(G, steps=7)
    # Cosine similarity between expected and computed, averaged over entries.
    cos = F.cosine_similarity(expected.flatten(), O.flatten(), dim=0)
    # NS with Keller's stability-tuned coefficients approximates U @ V.T
    # closely but not exactly (cos ~0.97 for a random 32x32 matrix).
    assert cos.item() > 0.96, f"NS does not match U@V.T: cos={cos.item():.4f}"


def test_newton_schulz_handles_tall_and_wide():
    for shape in [(32, 128), (128, 32), (16, 16)]:
        G = torch.randn(*shape)
        O = zeropower_via_newton_schulz(G, steps=5)
        assert O.shape == G.shape
        s = torch.linalg.svdvals(O)
        assert s.min() > 0.5
        assert s.max() < 1.5


# -----------------------------------------------------------------------------
# Muon step
# -----------------------------------------------------------------------------
def test_muon_step_rejects_1d_params():
    p = nn.Parameter(torch.randn(8))
    opt = Muon([p], lr=1e-2)
    p.grad = torch.randn_like(p)
    with pytest.raises(ValueError):
        opt.step()


def test_muon_reduces_convex_loss():
    """On a simple quadratic, Muon should reduce loss over steps."""
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(32, 24))
    target = torch.randn(32, 24)
    opt = Muon([W], lr=0.1, momentum=0.9)

    initial_loss = ((W - target) ** 2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        loss = ((W - target) ** 2).mean()
        loss.backward()
        opt.step()
    final_loss = ((W - target) ** 2).mean().item()
    assert final_loss < initial_loss * 0.5, (
        f"Muon did not reduce loss enough: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_muon_momentum_state_persists():
    """Momentum buffer should accumulate across steps."""
    W = nn.Parameter(torch.randn(8, 8))
    opt = Muon([W], lr=1e-3, momentum=0.9)
    W.grad = torch.randn_like(W)
    opt.step()
    buf1 = opt.state[W]["momentum_buffer"].clone()
    W.grad = torch.randn_like(W)
    opt.step()
    buf2 = opt.state[W]["momentum_buffer"]
    assert not torch.allclose(buf1, buf2), "momentum buffer did not update"


# -----------------------------------------------------------------------------
# Hybrid optimiser param routing
# -----------------------------------------------------------------------------
def _small_hydra_model() -> HydraLM:
    cfg = HydraConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2,
        swa_every=2, swa_window=16, dn_chunk_size=8,
        max_position_embeddings=128,
    )
    return HydraLM(cfg)


def test_hybrid_optimizer_routes_params_correctly():
    model = _small_hydra_model()
    opt = build_hybrid_optimizer(model, muon_lr=1e-2, adamw_lr=3e-4)
    assert isinstance(opt, HybridMuonAdamW)

    muon_params = {id(p) for g in opt._muon.param_groups for p in g["params"]}
    adamw_params = {id(p) for g in opt._adamw.param_groups for p in g["params"]}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            assert id(p) in adamw_params, f"1D param {name} leaked to Muon"
        elif "embed" in name.lower() or "lm_head" in name.lower():
            assert id(p) in adamw_params, f"embed/head {name} should be AdamW"
        else:
            assert id(p) in muon_params, f"hidden matrix {name} should be Muon"

    # Every trainable param is routed exactly once.
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(muon_params) + len(adamw_params) == n_params


def test_hybrid_optimizer_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = _small_hydra_model()
    opt = build_hybrid_optimizer(model, muon_lr=2e-2, adamw_lr=3e-3)

    ids = torch.randint(0, 64, (2, 16))
    labels = torch.randint(0, 64, (2, 16))

    def step() -> float:
        logits = model(ids)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, 64), labels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    l0 = step()
    for _ in range(20):
        last = step()
    assert last < l0, f"hybrid Muon+AdamW did not reduce training loss: {l0:.4f} -> {last:.4f}"
    assert math.isfinite(last)


def test_hybrid_optimizer_state_dict_roundtrip():
    model = _small_hydra_model()
    opt = build_hybrid_optimizer(model)
    # Do one step so state is populated.
    ids = torch.randint(0, 64, (1, 8))
    logits = model(ids)["logits"]
    loss = logits.mean()
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    assert "muon" in sd and "adamw" in sd

    # Reload into a fresh optimiser.
    model2 = _small_hydra_model()
    # Match param identity by loading weights too.
    model2.load_state_dict(model.state_dict())
    opt2 = build_hybrid_optimizer(model2)
    opt2.load_state_dict(sd)
