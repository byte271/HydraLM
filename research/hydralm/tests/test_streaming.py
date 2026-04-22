"""
Streaming correctness: chunk-wise forward must equal single-shot forward.

This is the mathematical backbone of the million-token claim.  If the
recurrent state is threaded correctly through every module (short
conv, Gated DeltaNet, Sliding-Window Attention), then processing a
sequence in chunks is bit-identical — up to fp32 noise — to processing
it all at once.

These tests exercise BOTH layer types (deltanet + SWA) and a range of
chunk sizes, including pathological ones (chunk 1, chunk that doesn't
divide N, chunk larger than the SWA window).
"""
from __future__ import annotations

import pytest
import torch

from hydralm import HydraConfig, HydraLM


def _make_model(seed: int = 0, **cfg_overrides) -> HydraLM:
    torch.manual_seed(seed)
    cfg = HydraConfig(
        vocab_size=257,
        d_model=64,
        n_layers=4,
        n_heads=4,
        dn_chunk_size=16,
        swa_window=32,
        swa_every=2,        # alternate DN / SWA so both are stressed
        **cfg_overrides,
    )
    return HydraLM(cfg).eval().to(torch.float64)


def _streamed_forward(model: HydraLM, ids: torch.Tensor, chunk: int) -> torch.Tensor:
    """Run `model` over `ids` in chunks of size `chunk`, return concatenated logits."""
    out_logits = []
    state: list[dict] | None = None
    for s in range(0, ids.shape[1], chunk):
        c = ids[:, s:s + chunk]
        out = model(c, state=state, return_state=True)
        state = out["state"]
        out_logits.append(out["logits"])
    return torch.cat(out_logits, dim=1)


@pytest.mark.parametrize("chunk", [1, 3, 7, 16, 32, 64])
def test_streaming_equals_full_forward(chunk: int) -> None:
    model = _make_model(seed=chunk)
    torch.manual_seed(42)
    ids = torch.randint(0, 257, (2, 96))

    with torch.no_grad():
        full = model(ids)["logits"]
        streamed = _streamed_forward(model, ids, chunk=chunk)

    torch.testing.assert_close(streamed, full, atol=1e-8, rtol=1e-6)


def test_streaming_beyond_swa_window() -> None:
    """Sequence much longer than SWA window: tests that SWA layers
    correctly bound their cache and still produce identical output."""
    model = _make_model(seed=7)
    torch.manual_seed(7)
    ids = torch.randint(0, 257, (1, 200))  # 200 > swa_window=32

    with torch.no_grad():
        full = model(ids)["logits"]
        streamed = _streamed_forward(model, ids, chunk=17)

    torch.testing.assert_close(streamed, full, atol=1e-8, rtol=1e-6)


def test_state_bytes_independent_of_stream_length() -> None:
    """Peak state memory must not grow with the total number of tokens."""
    from hydralm.streaming import StreamingEngine

    model = _make_model(seed=1)
    engine = StreamingEngine(model, chunk_size=32, dtype=torch.float64)

    short = torch.randint(0, 257, (1, 64))
    long = torch.randint(0, 257, (1, 64 * 50))    # 50x longer

    s1 = engine.process(short)
    s2 = engine.process(long)

    # Peak memory must be identical (SWA cache is capped at `window` tokens,
    # DeltaNet state is fixed-shape). Allow a couple bytes of slack for
    # dictionary overhead differences but it should be exact in practice.
    assert s2.peak_state_bytes == s1.peak_state_bytes, (
        f"state grew from {s1.peak_state_bytes} to {s2.peak_state_bytes}"
    )
    assert s2.tokens_processed == 50 * s1.tokens_processed
