# Claims

This document enumerates every *quantitative* claim HydraLM makes. Every
claim is mechanically verifiable by a script under `scripts/`, gated in
CI at `https://github.com/byte271/hydralm`, and accompanied by a pass
threshold. Claims that cannot be verified automatically are *not*
listed here.

The philosophy is the one advocated by the "Reproducible ML" thread:
a research artifact should make it easy for a reader to say "is this
true?" and get a yes/no answer without reading the paper first.

## How to run every claim

```bash
python scripts/reproduce_claims.py --model checkpoints/step-50000
```

The script produces a pass/fail table and writes the raw numbers to
`results/claims.json`. CI fails on any regression.

---

## Correctness claims

### C1 — Training/inference equivalence

**Claim.** For any input sequence of length N, calling
`model(ids)` in parallel once and calling the model step-by-step
(feeding one token at a time and threading the state dict through
each call) produces the same final logits.

**Pass threshold.** `max |training_logits - streaming_logits| < 1e-4`
in bfloat16, `< 1e-6` in fp32.

**Verified by.** `tests/test_equivalence.py::test_streaming_matches_parallel`.

### C2 — No state leaks across batches

**Claim.** Stateful generation does not carry state between independent
batch elements when the user calls `generate(..., state=None)`.

**Pass threshold.** Pairwise logits across batch members differ if and
only if the inputs differ.

**Verified by.** `tests/test_shapes.py::test_batched_state_independence`.

### C3 — Fact-bank round-trip is lossless

**Claim.** For any `(key, value)` pair written to a non-full fact bank,
`bank.query(key, k=1)` returns the same value with cosine similarity
1.0 (up to fp32 precision).

**Pass threshold.** `|cos(v_returned, v_written) - 1| < 1e-6`.

**Verified by.** `tests/test_fact_bank.py::test_write_read_roundtrip`.

---

## Efficiency claims

### C4 — Inference memory is constant in sequence length

**Claim.** At inference time, the peak GPU memory is independent of
the number of tokens already generated, up to a multiplicative
constant that depends only on `cfg.swa_window`, `cfg.n_heads`,
`cfg.head_dim`, and `cfg.n_layers`.

**Pass threshold.** `mem(N=1M) / mem(N=4k) < 1.05`.

**Verified by.** `scripts/million_token_demo.py` + `tests/test_complexity.py`.

### C5 — Long-context throughput is constant

**Claim.** Per-token generation latency at `N = 1M` is within 10% of
the latency at `N = 4k`.

**Pass threshold.** `latency(N=1M) / latency(N=4k) < 1.10`.

**Verified by.** `scripts/benchmark_length.py`.

### C6 — Muon+AdamW is faster than AdamW at matched loss

**Claim.** Reaching a target validation loss of 3.00 nats on
TinyShakespeare takes fewer wall-clock minutes with
`HybridMuonAdamW` than with pure AdamW, same LR schedule.

**Pass threshold.** `t_muon / t_adamw < 0.85`.

**Verified by.** `tests/test_muon.py::test_wallclock_advantage`.

---

## Quality claims

### C7 — MQAR @ 256 pairs, 8k context

**Claim.** At `n_pairs = 256`, `seq_len = 8192`, HydraLM 4:1 160M
achieves exact-match accuracy ≥ 0.95.

**Pass threshold.** `acc >= 0.95`.

**Verified by.** `scripts/run_mqar.py --n-pairs 256`.

### C8 — Needle-in-a-Haystack @ 256k

**Claim.** At `N = 256k` tokens, needle depth 0.5, HydraLM 4:1 160M
retrieves the needle with accuracy ≥ 0.90.

**Pass threshold.** `acc >= 0.90`.

**Verified by.** `scripts/needle_in_haystack.py --max-length 262144`.

### C9 — Fact-bank p@1 after 10× rotations

**Claim.** After writing 1M random pairs to a 10k-capacity fact bank
(10× rotations), `p@1` on the most-recently-written 10% of pairs is
≥ 0.95.

**Pass threshold.** `p@1 >= 0.95`.

**Verified by.** `scripts/online_learning_demo.py --capacity 10000
--writes 1000000`.

---

## Non-claims (deliberately not made)

The following are *not* claims HydraLM makes, and should not be
expected to hold:

- **SOTA on a specific downstream benchmark.** HydraLM is a reference
  implementation, not a model card. If you need SOTA numbers on
  GLUE/MMLU/HumanEval, fine-tune a HydraLM checkpoint yourself and
  publish it separately.
- **Scaling beyond 1B parameters.** Nothing in the code prevents it,
  but we have not tested it and do not claim the numbers in
  `docs/benchmarks.md` transfer.
- **Arbitrary-length fact-bank recall without interference.** The
  claim is `p@1 >= 0.95` at 10× rotations; beyond that, accuracy
  degrades log-linearly.
- **Compatibility with flash-attention 3.** HydraLM's SWA uses a
  plain PyTorch implementation; FA3 hooks are on the roadmap but not
  promised.

If you want a new claim added, please open a PR at
`https://github.com/byte271/hydralm` that adds both the script that
verifies it and the threshold that separates pass from fail.
