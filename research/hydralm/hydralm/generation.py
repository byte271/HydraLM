"""
Streaming autoregressive generation for HydraLM.

Because DeltaNet layers maintain only a fixed-size recurrent state and
SWA layers maintain only a fixed-size ring buffer, the memory required
for generation is CONSTANT in the number of generated tokens — in
stark contrast to standard Transformer KV caches, which grow linearly
with the context length.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .model import HydraLM


@torch.no_grad()
def generate(
    model: HydraLM,
    prompt_ids: torch.Tensor,           # (B, N0)
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressive generation with O(1) per-step memory.

    Returns generated token ids of shape (B, N0 + T) where T <= max_new_tokens.
    The prompt prefix is always preserved verbatim.

    Args:
        model:          HydraLM (set to eval before calling).
        prompt_ids:     (B, N0) — integer token ids.
        max_new_tokens: maximum number of new tokens to generate.
        temperature:    sampling temperature (0 = greedy argmax).
        top_k:          if > 0, restrict to the top-k logits before sampling.
        top_p:          nucleus (top-p) probability threshold in (0, 1).
        eos_token_id:   if set, stop once every sequence in the batch has
                        generated this token id.

    Notes
    -----
    * ``top_k`` and ``top_p`` are applied in that order.
    * ``temperature=0`` uses greedy argmax regardless of ``top_k`` / ``top_p``.
    * When ``eos_token_id`` is not None, finished sequences continue to emit
      ``eos_token_id`` tokens so the output tensor is rectangular; callers
      that need variable-length output should strip the EOS suffix.
    """
    model.eval()
    device = prompt_ids.device
    B = prompt_ids.shape[0]

    # ------------------------------------------------------------------
    # 1) Prefill: one parallel forward over the whole prompt to build the
    #    initial recurrent state and grab the first decoding logits.
    # ------------------------------------------------------------------
    out = model(prompt_ids, return_state=True)
    state = out["state"]
    next_logits = out["logits"][:, -1, :]                   # (B, V)

    generated: list[torch.Tensor] = [prompt_ids]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # 2) Token-by-token decoding using the O(1) recurrent step.
    # ------------------------------------------------------------------
    for _ in range(max_new_tokens):
        next_token = _sample(next_logits, temperature, top_k, top_p)  # (B,)

        # Finished sequences emit EOS (or 0) so the output is rectangular.
        if eos_token_id is not None:
            pad = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(finished, pad, next_token)

        generated.append(next_token.unsqueeze(1))

        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

        # Step all sequences regardless of `finished`; the logits for
        # finished sequences will be masked on the next iteration.
        next_logits, state = model.step(next_token, state)

    return torch.cat(generated, dim=1)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample(
    logits: torch.Tensor,      # (B, V)
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:             # (B,)
    """Apply temperature / top-k / top-p filtering and sample one token per row."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        # Keep only the top-k logits; zero out the rest.
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k=k, dim=-1)
        # v[:, -1] is the k-th largest value per row.
        logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        # Nucleus sampling (Holtzman et al., 2020).
        # Sort descending, compute cumulative probability, and mask tokens
        # whose cumulative probability exceeds `top_p`.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        # Convert to probabilities in the sorted order.
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # Shift cumprobs right by one so that the token that pushes the
        # cumulative probability over top_p is still included.
        # (Without the shift the highest-probability token can be dropped
        # when top_p is tiny.)
        shifted = torch.cat(
            [torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]], dim=-1
        )
        to_remove = shifted > top_p
        sorted_logits.masked_fill_(to_remove, float("-inf"))
        # Scatter back into the original token ordering.
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
