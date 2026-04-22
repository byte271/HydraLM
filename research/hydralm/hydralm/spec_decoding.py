"""
Speculative decoding for HydraLM.

Given a fast *draft* model and a larger *target* model that share a vocabulary,
speculative decoding generates ``k`` tokens with the draft, then runs the
target ONCE in parallel to either accept or reject each draft token — yielding
the EXACT target-model distribution but at ``1 + accepted_len`` target tokens
per target forward pass.

Algorithm (Chen et al., 2023 / Leviathan et al., 2023)
-------------------------------------------------------
Let p(.) be the target distribution and q(.) be the draft distribution at
position t.

  1. Draft proposes x_{t+1}, ..., x_{t+k} by sampling from q.
  2. Target is run on the k drafted tokens IN PARALLEL, producing
     p_{t+1}, ..., p_{t+k+1}.
  3. For each drafted x_{t+i}, accept with probability
         min(1, p(x_{t+i}) / q(x_{t+i})).
  4. On the first rejection at position j, sample a replacement from the
     residual distribution max(0, p - q) / Z and stop.
  5. If all k drafts were accepted, sample one bonus token from p_{t+k+1}.

This is *mathematically exact* — the output distribution matches the target
model token-for-token.

HydraLM & state rollback
------------------------
On rejection we must roll BOTH models' states back to the beginning of the
round, then advance them by exactly (accepted_prefix + resample). To enable
this, we clone both states at the start of each round. Cloning is O(layers
× state_size) and independent of sequence length — HydraLM's entire design
advantage flowing through.

Batch semantics
---------------
Per-row acceptance is tracked independently. The round terminates when
the MINIMUM accepted length across the batch is reached, keeping both
models' states in sync without per-row bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import HydraLM


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodingStats:
    """Aggregate statistics for a single speculative-decoding call."""
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def mean_tokens_per_round(self) -> float:
        # Each round commits accepted_len + 1 tokens.
        return (self.accepted + self.rounds) / max(1, self.rounds)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_probs(
    logits: Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tensor:
    """Apply temperature / top-k / top-p and return a normalised probability
    distribution over the vocabulary.

    ``temperature <= 0`` returns a one-hot argmax distribution so that greedy
    decoding is treated identically to sampling in the acceptance test.
    """
    if temperature <= 0.0:
        idx = logits.argmax(dim=-1, keepdim=True)
        out = torch.zeros_like(logits)
        out.scatter_(-1, idx, 1.0)
        return out

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k=k, dim=-1)
        logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # Shift right so the token that pushes the cumulative sum over top_p
        # is still included (avoids zeroing the highest-probability token).
        shifted = torch.cat(
            [torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]], dim=-1
        )
        sorted_logits.masked_fill_(shifted > top_p, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    return F.softmax(logits, dim=-1)


def _clone_state(state: Optional[list[dict]]) -> Optional[list[dict]]:
    """Deep-clone a HydraLM state list. Tensors are cloned; scalars copied."""
    if state is None:
        return None
    out: list[dict] = []
    for layer in state:
        if layer is None:
            out.append({})
            continue
        new_layer: dict = {}
        for k, v in layer.items():
            new_layer[k] = v.clone() if torch.is_tensor(v) else v
        out.append(new_layer)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_generate(
    target: HydraLM,
    draft: HydraLM,
    prompt_ids: Tensor,                     # (B, N0) int64
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> tuple[Tensor, SpecDecodingStats]:
    """Exact speculative decoding with HydraLM's recurrent state.

    Returns
    -------
    (generated_ids, stats) where ``generated_ids`` has shape
    ``(B, N0 + max_new_tokens)`` (the output is always trimmed to exactly
    ``prompt_ids.shape[1] + max_new_tokens`` tokens).

    Notes
    -----
    * ``target`` and ``draft`` must share ``vocab_size``.
    * The batch round terminates at the MIN accepted length across all rows,
      keeping both models' states in sync without per-row bookkeeping.
    * Both models are set to ``eval()`` at the start of the call.
    """
    assert target.cfg.vocab_size == draft.cfg.vocab_size, (
        f"vocab mismatch: target={target.cfg.vocab_size}, draft={draft.cfg.vocab_size}"
    )
    target.eval()
    draft.eval()
    device = prompt_ids.device
    B = prompt_ids.shape[0]

    # ------------------------------------------------------------------
    # Prefill both models on the prompt.
    # ------------------------------------------------------------------
    t_out = target(prompt_ids, return_state=True)
    d_out = draft(prompt_ids, return_state=True)
    t_state: list[dict] = t_out["state"]
    d_state: list[dict] = d_out["state"]
    # Logits at the last prompt position (used to seed the first decoding round).
    last_target_logits: Tensor = t_out["logits"][:, -1, :]   # (B, V)
    d_last_logits: Tensor = d_out["logits"][:, -1, :]        # (B, V)

    generated: list[Tensor] = [prompt_ids]
    produced = 0
    stats = SpecDecodingStats()
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    while produced < max_new_tokens:
        stats.rounds += 1

        # Snapshot both states so we can roll back on any rejection.
        t_state_pre = _clone_state(t_state)
        d_state_pre = _clone_state(d_state)

        # ------------------------------------------------------------------
        # Step 1: Draft proposes k tokens.
        # ------------------------------------------------------------------
        draft_tokens: list[Tensor] = []
        q_probs: list[Tensor] = []
        cur_d_logits = d_last_logits
        for _ in range(k):
            q = _sample_probs(cur_d_logits, temperature, top_k, top_p)
            x = torch.multinomial(q, 1).squeeze(-1)
            draft_tokens.append(x)
            q_probs.append(q)
            cur_d_logits, d_state = draft.step(x, d_state)

        draft_ids = torch.stack(draft_tokens, dim=1)         # (B, k)

        # ------------------------------------------------------------------
        # Step 2: Target runs IN PARALLEL on the k draft tokens.
        #
        # We feed `draft_ids` through the target starting from the PRE-round
        # snapshot so that the target sees tokens [t+1 .. t+k] in context
        # after the prompt+all previously committed tokens.
        # ------------------------------------------------------------------
        t_par = target(draft_ids, state=t_state_pre, return_state=True)
        target_logits_seq: Tensor = t_par["logits"]          # (B, k, V)

        # p_logits_list[i] are the target logits used to evaluate
        # draft token i.  p_logits_list[0] is the last pre-round target
        # logit (conditioned on everything before this round).
        p_logits_list: list[Tensor] = [last_target_logits] + [
            target_logits_seq[:, i, :] for i in range(k - 1)
        ]
        p_last_after_k: Tensor = target_logits_seq[:, -1, :]

        # ------------------------------------------------------------------
        # Step 3: Acceptance test — token-level, per batch row.
        # ------------------------------------------------------------------
        # accept_len[b] = how many tokens row b accepted (0 .. k).
        accept_len = torch.full((B,), k, dtype=torch.long, device=device)

        for i in range(k):
            p = _sample_probs(p_logits_list[i], temperature, top_k, top_p)
            q = q_probs[i]
            x_i = draft_ids[:, i]                           # (B,)
            p_xi = p.gather(1, x_i.unsqueeze(1)).squeeze(1)  # (B,)
            q_xi = q.gather(1, x_i.unsqueeze(1)).squeeze(1)  # (B,)
            ratio = (p_xi / q_xi.clamp_min(1e-20)).clamp_max(1.0)
            u = torch.rand_like(ratio)
            accepted_here = u < ratio
            stats.proposed += int(accepted_here.numel())
            # A row can reject at position i only if it has not yet rejected
            # at any earlier position, i.e. accept_len[b] == k (still at
            # the original full-accept value).  Using `> i` instead of
            # `>= i` correctly identifies rows that have not yet rejected
            # (they still hold accept_len == k, which is > i for all i < k).
            not_yet_rejected = accept_len > i
            newly_rejected = (~accepted_here) & not_yet_rejected
            accept_len = torch.where(
                newly_rejected, torch.full_like(accept_len, i), accept_len
            )

        # Use the minimum accepted length across the batch to keep states
        # in sync across rows.
        round_len = int(accept_len.min().item())
        stats.accepted += round_len * B

        # ------------------------------------------------------------------
        # Step 4: Commit tokens and re-synchronise both models' states.
        # ------------------------------------------------------------------
        if round_len == k:
            # All k draft tokens accepted — emit them plus one bonus token.
            bonus_probs = _sample_probs(p_last_after_k, temperature, top_k, top_p)
            bonus = torch.multinomial(bonus_probs, 1).squeeze(-1)           # (B,)
            committed = torch.cat([draft_ids, bonus.unsqueeze(1)], dim=1)   # (B, k+1)

            # Target state: the parallel forward gave us the state after
            # processing the k draft tokens; step once more on the bonus.
            t_state = t_par["state"]
            last_target_logits, t_state = target.step(bonus, t_state)

            # Draft state: already post-k drafts from the step loop; advance
            # by the bonus token.
            d_last_logits, d_state = draft.step(bonus, d_state)

        else:
            # First rejection at position `round_len`.
            accepted_ids = draft_ids[:, :round_len]                         # (B, round_len)
            # Residual distribution: max(0, p - q) / Z.
            p_at_rej = _sample_probs(p_logits_list[round_len], temperature, top_k, top_p)
            q_at_rej = q_probs[round_len]
            resid = (p_at_rej - q_at_rej).clamp_min(0.0)
            resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            resample = torch.multinomial(resid, 1).squeeze(-1)              # (B,)
            committed = torch.cat([accepted_ids, resample.unsqueeze(1)], dim=1)  # (B, round_len+1)

            # Rebuild both states from the pre-round snapshot by running
            # the committed tokens through both models in parallel.
            t_sync = target(committed, state=t_state_pre, return_state=True)
            t_state = t_sync["state"]
            last_target_logits = t_sync["logits"][:, -1, :]

            d_sync = draft(committed, state=d_state_pre, return_state=True)
            d_state = d_sync["state"]
            d_last_logits = d_sync["logits"][:, -1, :]

        generated.append(committed)
        produced += committed.shape[1]

        if eos_token_id is not None:
            finished = finished | (committed == eos_token_id).any(dim=1)
            if finished.all():
                break

    out = torch.cat(generated, dim=1)
    # Trim to exactly prompt_len + max_new_tokens.
    target_len = prompt_ids.shape[1] + max_new_tokens
    if out.shape[1] > target_len:
        out = out[:, :target_len]
    return out, stats


__all__ = ["speculative_generate", "SpecDecodingStats"]
