"""
Muon optimizer — "MomentUm Orthogonalised by Newton-schulz".

Reference: Keller Jordan, "Muon: An optimizer for the hidden layers of neural
networks" (2024). The idea: maintain an EMA momentum buffer `B` per 2D weight
matrix, then apply the orthogonalised update

    W <- W - lr * O(B)

where O(.) is a matrix function that maps `B` to the nearest semi-orthogonal
matrix (in Frobenius norm). `O` is approximated in 5 Newton-Schulz iterations
using the odd polynomial p(x) = a*x + b*x^3 + c*x^5 with carefully-tuned
coefficients that keep every singular value in [0, ~1.2] throughout the
iteration and converge to the sign function on (0, sigma_max].

This is ~1.3x more sample-efficient than AdamW at matched FLOPs on language
model pretraining at the scales tested by Jordan, Essential AI, and Kimi.

Scope of application
--------------------
Muon is defined for 2D weight matrices only. Embeddings, biases, RMSNorm gains,
and the LM head must use a scalar optimiser (AdamW). `build_hybrid_optimizer`
does this split automatically and returns a ``HybridMuonAdamW`` that exposes a
single ``step()`` / ``zero_grad()`` interface compatible with the HydraLM
``Trainer`` and ``torch.optim.lr_scheduler``.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz iteration
# ---------------------------------------------------------------------------
# Jordan's coefficients. The triple (3.4445, -4.7750, 2.0315) gives a quintic
# that on [0, 1] maps every singular value upward toward 1, with maximum
# deviation < 0.3 after 5 iterations — enough for an "orthogonalising"
# preconditioner while remaining cheap on GPUs/CPUs.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Return an approximation of U @ V.T where G = U @ S @ V.T is the SVD of G.

    For a 2D matrix, this is the closest semi-orthogonal matrix to G (it
    replaces every singular value with 1 while keeping left/right singular
    vectors). Batched across any leading dims.
    """
    assert G.ndim >= 2, "Muon orthogonalisation expects 2D+ tensors"
    a, b, c = _NS_COEFFS
    X = G.to(torch.float32)
    # Transpose tall matrices so the NS iteration runs on the smaller dim^2.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    # Normalise to spectral norm <= 1 so the fixed-point iteration is stable.
    # Frobenius is a cheap upper bound on the spectral norm.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimiser
# ---------------------------------------------------------------------------
class Muon(Optimizer):
    """Muon optimiser for 2D+ weight matrices.

    Args:
        params:        iterable of 2D+ parameters (others must go to AdamW).
        lr:            base learning rate.
        momentum:      EMA coefficient for the momentum buffer (default 0.95).
        nesterov:      if True, apply a Nesterov look-ahead (recommended).
        ns_steps:      number of Newton-Schulz iterations (default 5).
        weight_decay:  decoupled weight decay (applied to the param itself).
        adjust_lr:     if True, scale lr by ``sqrt(max(1, fan_out/fan_in))``
                       so that the effective step size is invariant to the
                       aspect ratio of the weight matrix (recommended).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 5e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adjust_lr: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            adjust_lr = group["adjust_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError(
                        "Muon received a <2D parameter. Route biases, norms, "
                        "and embeddings to AdamW via build_hybrid_optimizer()."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if nesterov else buf

                # Collapse any leading dims to a 2D view for orthogonalisation.
                view_shape = update.shape
                u2d = update.reshape(view_shape[0], -1)
                u2d = zeropower_via_newton_schulz(u2d, steps=ns_steps)
                update = u2d.view(view_shape)

                # Aspect-ratio LR correction — keeps effective update size
                # comparable across rectangular matrices.
                if adjust_lr:
                    fan_out, fan_in = view_shape[0], u2d.shape[-1]
                    scale = max(1.0, fan_out / max(1, fan_in)) ** 0.5
                    step_lr = lr * scale
                else:
                    step_lr = lr

                if wd != 0.0:
                    p.mul_(1.0 - step_lr * wd)
                p.add_(update, alpha=-step_lr)

        return loss


# ---------------------------------------------------------------------------
# Hybrid wrapper: Muon for 2D matrices, AdamW for everything else.
# ---------------------------------------------------------------------------
class HybridMuonAdamW(Optimizer):
    """Composite optimiser routing 2D matrices to Muon and scalars to AdamW.

    This class exposes a single ``step()`` / ``zero_grad()`` interface so that
    the outside world (schedulers, checkpointing, grad scalers) sees a single
    optimiser.

    ``param_groups`` is the concatenation of both inner optimisers' groups so
    that ``LambdaLR`` can scale both learning rates uniformly.

    The ``state`` property delegates to both inner optimisers so that
    ``state_dict()`` / ``load_state_dict()`` and per-parameter state
    inspections (e.g. by grad scalers) work correctly.
    """

    def __init__(self, muon: Muon, adamw: Optimizer) -> None:
        self._muon = muon
        self._adamw = adamw
        # Combine param groups so LambdaLR sees everything.
        # We do NOT call super().__init__ — instead we manually set the
        # attributes that Optimizer subclasses and schedulers inspect.
        self.defaults: dict = {}
        self.param_groups = list(muon.param_groups) + list(adamw.param_groups)

    # ------------------------------------------------------------------
    # State delegation: merge both inner state dicts into a single view.
    # torch.optim.lr_scheduler and grad scalers look up ``optimizer.state``
    # by parameter identity, so we must expose a unified mapping.
    # ------------------------------------------------------------------
    @property
    def state(self):  # type: ignore[override]
        """Read-only merged view of both inner optimisers' per-parameter state."""
        merged: dict = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    @state.setter
    def state(self, value: dict) -> None:
        # Called by Optimizer.__init__ with an empty dict; we accept but
        # discard it since we manage state via the inner optimisers.
        pass

    # ------------------------------------------------------------------
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:  # type: ignore[override]
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, sd: dict) -> None:  # type: ignore[override]
        self._muon.load_state_dict(sd["muon"])
        self._adamw.load_state_dict(sd["adamw"])
        # Keep param_groups in sync after loading (schedulers may have
        # mutated the lr stored inside them).
        self.param_groups = list(self._muon.param_groups) + list(self._adamw.param_groups)


# ---------------------------------------------------------------------------
# Parameter split helper
# ---------------------------------------------------------------------------
def _should_use_muon(name: str, p: Tensor) -> bool:
    """Route 2D+ weight matrices in hidden layers to Muon.

    Explicitly excluded:
      * parameters with ndim < 2 (biases, RMSNorm gains, 1D gates),
      * token embeddings (``embed`` / ``wte`` / ``word_embeddings``),
      * the LM head (``lm_head``) — matches Jordan's original split.
    """
    if p.ndim < 2:
        return False
    lname = name.lower()
    if "embed" in lname or "lm_head" in lname or "wte" in lname:
        return False
    return True


def build_hybrid_optimizer(
    model: torch.nn.Module,
    *,
    muon_lr: float = 5e-3,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_weight_decay: float = 0.1,
    adamw_eps: float = 1e-8,
    fused: bool | None = None,
) -> HybridMuonAdamW:
    """Split parameters and return a ready-to-use hybrid optimiser.

    The Muon group uses ``weight_decay=0`` by default (the orthogonalised
    update is already normalised); the AdamW group uses 0.1 weight decay on
    2D+ scalar-like params and 0 on biases/norms.
    """
    muon_params: list[Tensor] = []
    adamw_decay: list[Tensor] = []
    adamw_nodecay: list[Tensor] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _should_use_muon(name, p):
            muon_params.append(p)
        elif p.ndim <= 1 or name.endswith(".bias"):
            adamw_nodecay.append(p)
        else:
            adamw_decay.append(p)

    if fused is None:
        fused = torch.cuda.is_available()

    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
    )
    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": adamw_weight_decay})
    if adamw_nodecay:
        adamw_groups.append({"params": adamw_nodecay, "weight_decay": 0.0})
    if not adamw_groups:
        # Minimal placeholder so AdamW initialises without error.
        adamw_groups.append({"params": [], "weight_decay": 0.0})
    adamw = torch.optim.AdamW(
        adamw_groups, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, fused=fused,
    )

    return HybridMuonAdamW(muon, adamw)


__all__ = [
    "Muon",
    "HybridMuonAdamW",
    "build_hybrid_optimizer",
    "zeropower_via_newton_schulz",
]
