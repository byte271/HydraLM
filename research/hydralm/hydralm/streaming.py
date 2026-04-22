"""
Million-token streaming engine.

HydraLM's recurrent form stores a *fixed-size* state per layer:

  * DeltaNet layer  : S ∈ R^{H × D × D}   and   conv_cache ∈ R^{3·inner × (K-1)}
  * SWA layer       : (k_cache, v_cache) ∈ R^{H × W × D}

None of these depend on the sequence length `N`.  This module exposes
a simple API that feeds an arbitrary-length token stream through the
model chunk by chunk, carrying state between chunks, so that peak
memory is O(state) — truly independent of `N`.

The mathematics is exact: streaming over chunks produces bit-identical
(up to fp32 numerical tolerance) output as running the full sequence
in one shot.  This is verified by `tests/test_streaming.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

from .model import HydraLM


@dataclass
class StreamStats:
    tokens_processed: int = 0
    chunks_processed: int = 0
    peak_state_bytes: int = 0
    last_state_bytes: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        mb = self.peak_state_bytes / (1024 ** 2)
        tok_per_s = self.tokens_processed / max(self.elapsed_seconds, 1e-9)
        return (
            f"processed {self.tokens_processed:,} tokens in "
            f"{self.chunks_processed:,} chunks "
            f"({tok_per_s:,.1f} tok/s)  |  "
            f"peak state {mb:.3f} MiB"
        )


def _state_bytes(state: list[dict]) -> int:
    """Compute the total number of bytes held by a HydraLM state list."""
    total = 0
    for layer_state in state:
        if layer_state is None:
            continue
        for v in layer_state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


class StreamingEngine:
    """Process arbitrarily long token streams with O(1) memory w.r.t. length.

    Example
    -------
    ::

        engine = StreamingEngine(model, chunk_size=1024)
        for logits, stats in engine.stream(token_iterator):
            ...   # logits are for the most recent chunk only

    The engine keeps the model in ``eval()`` mode and always operates
    inside ``torch.no_grad()``.  Use :meth:`process` for prefill-only
    (no logits retained) and :meth:`extend_and_generate` for the
    prefill-then-decode pipeline.
    """

    def __init__(
        self,
        model: HydraLM,
        chunk_size: int = 1024,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        keep_logits: bool = False,
    ) -> None:
        self.model = model.to(device=device, dtype=dtype).eval()
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.keep_logits = keep_logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, tokens: torch.Tensor, progress_every: int = 0) -> StreamStats:
        """Process a (B, N) token tensor, returning statistics only.

        Memory usage: O(state) — logits for each chunk are discarded.
        """
        assert tokens.dim() == 2, "expected (B, N)"
        N = tokens.shape[1]
        stats = StreamStats()
        state: Optional[list[dict]] = None
        t0 = time.perf_counter()

        for start in range(0, N, self.chunk_size):
            chunk = tokens[:, start:start + self.chunk_size].to(self.device)
            out = self.model(chunk, state=state, return_state=True)
            state = out["state"]                         # carry forward
            # Drop the logits reference so the memory can be reclaimed.
            del out
            stats.tokens_processed += chunk.shape[1]
            stats.chunks_processed += 1
            b = _state_bytes(state)
            stats.last_state_bytes = b
            stats.peak_state_bytes = max(stats.peak_state_bytes, b)
            if progress_every and stats.chunks_processed % progress_every == 0:
                elapsed = time.perf_counter() - t0
                tps = stats.tokens_processed / max(elapsed, 1e-9)
                print(
                    f"  [{stats.tokens_processed:>10,} tok | "
                    f"{stats.chunks_processed:>6} chunks | "
                    f"{tps:>8,.1f} tok/s | "
                    f"state {stats.last_state_bytes / 1024 ** 2:.3f} MiB]"
                )

        stats.elapsed_seconds = time.perf_counter() - t0
        return stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream(
        self,
        token_chunks: Iterator[torch.Tensor],
    ) -> Iterator[tuple[torch.Tensor, StreamStats]]:
        """Stream over arbitrary (B, N_i) chunks, yielding per-chunk logits.

        The model state persists between yields.  The caller is free to
        discard logits to keep memory bounded.
        """
        stats = StreamStats()
        state: Optional[list[dict]] = None
        t0 = time.perf_counter()

        for chunk in token_chunks:
            chunk = chunk.to(self.device)
            out = self.model(chunk, state=state, return_state=True)
            state = out["state"]
            logits = out["logits"]
            stats.tokens_processed += chunk.shape[1]
            stats.chunks_processed += 1
            b = _state_bytes(state)
            stats.last_state_bytes = b
            stats.peak_state_bytes = max(stats.peak_state_bytes, b)
            stats.elapsed_seconds = time.perf_counter() - t0
            yield logits, stats
            del logits, out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_and_generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Process a (possibly million-token) prompt in streaming chunks,
        then continue token-by-token decoding — O(1) memory per new token.

        Args:
            prompt:         (B, N0) integer token ids.
            max_new_tokens: number of new tokens to decode.
            temperature:    sampling temperature (0 = greedy argmax).
            top_k:          if given, restrict sampling to the top-k logits.

        Returns:
            (B, max_new_tokens) int64 tensor of generated tokens (prompt
            not included).
        """
        assert prompt.dim() == 2, "prompt must be 2D (B, N)"
        if prompt.shape[1] == 0:
            raise ValueError("prompt must contain at least one token")

        state: Optional[list[dict]] = None
        last_logits: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # 1. Stream-prefill the prompt chunk by chunk.
        # ------------------------------------------------------------------
        for start in range(0, prompt.shape[1], self.chunk_size):
            chunk = prompt[:, start:start + self.chunk_size].to(self.device)
            out = self.model(chunk, state=state, return_state=True)
            state = out["state"]
            last_logits = out["logits"][:, -1, :]        # (B, V) — last token

        # After the loop, `last_logits` is guaranteed to be assigned because
        # we validated prompt.shape[1] >= 1 above.
        assert last_logits is not None

        # ------------------------------------------------------------------
        # 2. Autoregressive decode using the O(1) recurrent step.
        # ------------------------------------------------------------------
        generated: list[torch.Tensor] = []
        cur_logits = last_logits

        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_tok = cur_logits.argmax(dim=-1)     # (B,)
            else:
                scaled = cur_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    v, _ = torch.topk(scaled, k)
                    scaled = scaled.masked_fill(
                        scaled < v[:, [-1]], float("-inf")
                    )
                probs = torch.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated.append(next_tok.unsqueeze(1))
            cur_logits, state = self.model.step(next_tok, state)

        return torch.cat(generated, dim=1)
