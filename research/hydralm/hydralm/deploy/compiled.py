"""
Compiled / batched decoding path for HydraLM.

This module provides a ``CompiledDecoder`` that wraps a ``HydraLM`` (or
``HydraLMForCausalLM``) for low-latency per-token generation:

  * ``torch.compile`` is applied to ``model.step`` with
    ``mode="reduce-overhead"`` and CUDA graphs, which is the setting that
    gives the biggest win for short, compute-light per-step kernels — and
    HydraLM's step IS exactly that (no KV-cache indexing, no O(N) SDP).

  * Request-level batching: each ``decode_batch`` call processes a list
    of in-flight sequences in one forward, advancing each sequence's
    recurrent state independently. Because HydraLM's state is a fixed-
    size tensor per sequence, we can trivially *concatenate* states
    across requests — no ragged KV-cache reshaping, which is the single
    biggest source of complexity in vLLM / TensorRT-LLM today.

The net effect: a Transformer server needs paged attention + KV cache
reshuffling to hit high batch utilisation; a HydraLM server gets it for
free because every sequence's state has the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hydralm.config import HydraConfig
from hydralm.model import HydraLM


@dataclass
class Request:
    """In-flight decoding request."""
    prompt: Tensor                     # (N0,) int64
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_token_id: Optional[int] = None
    produced: list[int] = field(default_factory=list)
    state: Optional[list] = None
    done: bool = False


class CompiledDecoder:
    """Low-latency batched decoder with torch.compile + CUDA graphs.

    Batching contract
    -----------------
    Requests in the same batch must have *equal prompt length* at prefill time.
    The Gated-DeltaNet recurrent state is shape-invariant across requests
    (``(H, D, D)`` regardless of N), but the sliding-window attention
    ``k_cache``/``v_cache`` are shape ``(H, min(N, window), D)`` — they only
    line up for stacking when prompts are bucketed by length (the pre-PagedAttention
    vLLM pattern). A paged-state follow-up would lift this restriction; it is
    intentionally out of scope for this revision.
    """

    def __init__(self, model: HydraLM, compile: bool = True):
        self.model = model.eval()
        if compile and hasattr(torch, "compile"):
            try:
                self._step = torch.compile(
                    model.step, mode="reduce-overhead", fullgraph=False, dynamic=False
                )
            except Exception:
                # compile can fail on older torch; fall back transparently
                self._step = model.step
        else:
            self._step = model.step

    # ---------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, reqs: list[Request]) -> None:
        """Run the parallel prefill for each request and stash its final state.

        Each request can have a different prompt length, so this is done one
        request at a time (the heavy cost is the *decode* loop, not prefill).
        """
        for r in reqs:
            out = self.model(r.prompt.unsqueeze(0), return_state=True)
            r.state = out["state"]
            r.produced = r.prompt.tolist()
            # Seed the sampler with the last prefill logits.
            r._last_logits = out["logits"][:, -1, :]  # (1, V)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def step_batch(self, reqs: list[Request]) -> None:
        """Advance one token for each not-done request.

        State concatenation: HydraLM per-layer state is a dict of tensors of
        identical shape across requests, so we can stack along batch dim,
        run a single ``step``, and re-split.
        """
        active = [r for r in reqs if not r.done]
        if not active:
            return

        # 1) Pack current last-logits into a (B, V) tensor.
        logits = torch.cat([r._last_logits for r in active], dim=0)

        # 2) Sample next tokens per request (temperature / top-k / top-p).
        next_tokens = _sample_batch(logits, active)  # (B,)

        # 3) Stack per-request state across batch.
        packed_state = _stack_states([r.state for r in active])

        # 4) Single fused step.
        logits_next, new_state = self._step(next_tokens, packed_state)

        # 5) Unpack and write back per-request.
        unpacked = _split_states(new_state, batch=len(active))
        for i, r in enumerate(active):
            tok = int(next_tokens[i].item())
            r.produced.append(tok)
            r.state = unpacked[i]
            r._last_logits = logits_next[i:i + 1]
            if r.eos_token_id is not None and tok == r.eos_token_id:
                r.done = True
            elif len(r.produced) - len(r.prompt) >= r.max_new_tokens:
                r.done = True

    # ---------------------------------------------------------------
    @torch.no_grad()
    def decode(self, reqs: list[Request]) -> list[Tensor]:
        """End-to-end: prefill + decode until all requests finish."""
        self.prefill(reqs)
        while any(not r.done for r in reqs):
            self.step_batch(reqs)
        return [torch.tensor(r.produced, dtype=torch.long) for r in reqs]


# ---------------------------------------------------------------------------
# State packing / unpacking helpers
# ---------------------------------------------------------------------------

def _stack_states(states: list[list[dict]]) -> list[dict]:
    """Stack per-layer dicts along batch dim. Expects identical keys/shapes."""
    n_layers = len(states[0])
    stacked: list[dict] = []
    for li in range(n_layers):
        layer_states = [s[li] for s in states]
        merged: dict = {}
        for k in layer_states[0].keys():
            vals = [ls[k] for ls in layer_states]
            if vals[0] is None:
                merged[k] = None
            elif isinstance(vals[0], torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            else:
                # Scalar (e.g. SWA pos counter) — take the first value.
                # All active requests share the same sequence length when
                # batched at equal prompt lengths, so the position counters
                # are identical.
                merged[k] = vals[0]
        stacked.append(merged)
    return stacked


def _split_states(stacked: list[dict], batch: int) -> list[list[dict]]:
    """Inverse of ``_stack_states``."""
    unpacked: list[list[dict]] = [[] for _ in range(batch)]
    for layer_state in stacked:
        per_req: list[dict] = [dict() for _ in range(batch)]
        for k, v in layer_state.items():
            if v is None:
                for i in range(batch):
                    per_req[i][k] = None
            elif isinstance(v, torch.Tensor):
                chunks = v.chunk(batch, dim=0)
                for i, c in enumerate(chunks):
                    per_req[i][k] = c
            else:
                for i in range(batch):
                    per_req[i][k] = v
        for i in range(batch):
            unpacked[i].append(per_req[i])
    return unpacked


# ---------------------------------------------------------------------------
# Per-request sampling
# ---------------------------------------------------------------------------

def _sample_one(lg: Tensor, r: Request) -> int:
    """Sample the next token for a single request.

    Applies temperature, top-k, and top-p (nucleus) filtering in that order.
    The top-p implementation correctly uses a right-shifted cumulative sum so
    that the highest-probability token is never erroneously masked.

    Args:
        lg: (V,) logit vector for one request.
        r:  the Request object carrying sampling hyperparameters.

    Returns:
        Sampled token id as a Python int.
    """
    if r.temperature <= 0.0:
        return int(lg.argmax().item())

    lg = lg / r.temperature

    if r.top_k is not None and r.top_k > 0:
        k = min(r.top_k, lg.size(-1))
        v, _ = torch.topk(lg, k=k)
        lg = lg.masked_fill(lg < v[-1], float("-inf"))

    if r.top_p is not None and 0.0 < r.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(lg, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # Shift right so the token that *crosses* top_p is still included.
        shifted = torch.cat([torch.zeros(1, device=lg.device), cumprobs[:-1]])
        sorted_logits = sorted_logits.masked_fill(shifted > r.top_p, float("-inf"))
        lg = torch.zeros_like(lg).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(lg, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _sample_batch(logits: Tensor, reqs: list[Request]) -> Tensor:
    """Sample one token per request in the batch.

    Each request may have different sampling hyperparameters so we loop
    per-request rather than vectorising (typical batch sizes at inference are
    small enough that the loop overhead is negligible).
    """
    out = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
    for i, r in enumerate(reqs):
        out[i] = _sample_one(logits[i], r)
    return out


__all__ = ["CompiledDecoder", "Request"]
