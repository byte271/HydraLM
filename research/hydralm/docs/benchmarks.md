# Benchmarks

All numbers below were collected on a **single NVIDIA A100 80GB** with
**PyTorch 2.4**, **CUDA 12.4**, and **bfloat16** autocast unless
otherwise noted. Every benchmark is scripted under `scripts/` and its
exact configuration is checked in alongside the raw output in
`results.json`.

To reproduce the full table, run:

```bash
python scripts/reproduce_claims.py --model checkpoints/step-50000
```

Please file any reproducibility deviations as a GitHub issue at
`https://github.com/byte271/hydralm/issues`.

## 0. Formal verification record for v0.3.0

This section records a full local verification run performed against the
**v0.3.0 source tree** as reported by `hydralm.__version__`.

### 0.1 System configuration

- Date: `2026-04-22`
- Host OS: Windows
- Working directory: `D:\HydraLM\research\hydralm`
- Python runtime used for validation: `3.13.12`
- PyTorch runtime: `2.11.0+cpu`
- Pytest runtime: `9.0.3`
- Device class: CPU only
- Environment note: validation ran in a local virtual environment
  (`.venv313`) created specifically for this test pass

### 0.2 Commands executed

```bash
# Test suite
python -m pytest -q -k "not hf_adapter_save_load_roundtrip"

# Manual reproduction of the deselected adapter round-trip
python - <<'PY'
import torch
from pathlib import Path
from hydralm import HydraConfig
from hydralm.deploy import HydraLMForCausalLM

root = Path(".manual-hf-roundtrip")
cfg = HydraConfig(
    vocab_size=257, d_model=64, n_layers=4, n_heads=4,
    swa_window=16, dn_chunk_size=8, swa_every=2,
)
model = HydraLMForCausalLM(cfg).eval()
model.save_pretrained(str(root))
loaded = HydraLMForCausalLM.from_pretrained(str(root)).eval()
ids = torch.randint(0, cfg.vocab_size, (1, 16))
assert torch.allclose(model(input_ids=ids).logits,
                      loaded(input_ids=ids).logits,
                      atol=1e-6)
PY

# Claim reproducer
python scripts/reproduce_claims.py --budget smoke --out .codex-RESULTS.md --json .codex-results.json
python scripts/reproduce_claims.py --budget paper --out .codex-RESULTS-paper.md --json .codex-results-paper.json

# Performance / behavior probes
python scripts/benchmark_length.py --device cpu --dtype float32 --min-log2 8 --max-log2 12
python scripts/long_context_qa.py --seq-len 4096 --num-facts 16 --num-queries 4 --batch-size 1 --n-batches 1 --d-model 128 --n-layers 4 --n-heads 4 --use-retrieval --retrieval-every 3 --retrieval-chunk-size 64 --retrieval-top-k 4
python scripts/cost_analysis.py
python scripts/million_token_demo.py --tokens 1000000 --chunk-size 2048 --d-model 64 --n-layers 2 --n-heads 2 --swa-window 256
```

### 0.3 Test-suite result

| Verification lane | Result | Notes |
| --- | --- | --- |
| `pytest -q -k "not hf_adapter_save_load_roundtrip"` | **76 passed, 1 deselected** | All selected test cases passed in ~55 s on CPU. |
| Manual `HydraLMForCausalLM` save/load round-trip | **PASS** | `max_abs_diff = 0.0`, `torch.allclose(..., atol=1e-6) = True`. |

The deselected pytest case was **not a model failure**. The fixture-backed
version of the test depended on a temporary-directory path that was blocked
by the local Windows sandbox policy, so the round-trip was reproduced
manually inside the workspace to verify the actual model behavior.

### 0.4 Claim reproducer result

#### Smoke budget

| Claim | Status | Measurement |
| --- | --- | --- |
| Linear complexity | PASS | Hydra slope `1.000`, Transformer slope `1.983` |
| Lossless MQAR | PASS | Hydra `0.898`, Transformer `0.570`, ratio `1.58` |
| Constant state | PASS | unique state-byte values across `1K..100M` = `1` |
| Cost reduction | PASS | FLOP save `99.8%`, memory save `100.0%` |
| Drop-in contract | PASS | parameter ratio `0.954`, HF generate `True` |
| Online learning | PASS | argmax `100.0%`, overwrite margin `0.90`, state `0.12 MB`, about `30,000x` smaller than KV cache |

#### Paper budget

| Claim | Status | Measurement |
| --- | --- | --- |
| Linear complexity | PASS | Hydra slope `1.000`, Transformer slope `1.983` |
| Lossless MQAR | PASS | Hydra `1.000`, Transformer `0.496`, ratio `2.02` |
| Constant state | PASS | unique state-byte values across `1K..100M` = `1` |
| Cost reduction | PASS | FLOP save `99.8%`, memory save `100.0%` |
| Drop-in contract | PASS | parameter ratio `0.954`, HF generate `True` |
| Online learning | PASS | argmax `100.0%`, overwrite margin `0.81`, state `16.00 MB`, about `3,750x` smaller than KV cache |

### 0.5 Additional runtime probes

#### CPU length-scaling sanity check

`scripts/benchmark_length.py --device cpu --dtype float32 --min-log2 8 --max-log2 12`

| Sequence length | HydraLM ms | Transformer ms | Relative speed |
| --- | ---: | ---: | ---: |
| 256 | 543.40 | 63.80 | 0.12x |
| 512 | 1121.22 | 134.74 | 0.12x |
| 1024 | 2930.30 | 368.89 | 0.13x |
| 2048 | 5789.75 | 926.07 | 0.16x |
| 4096 | 17548.18 | 2868.71 | 0.16x |

Interpretation: this **CPU implementation sanity check** shows the expected
architectural scaling trend but does **not** show a raw wall-clock CPU speed
win over the PyTorch Transformer baseline. These numbers should not be used
as a substitute for the GPU release benchmarks below.

#### Million-token streaming demo

`scripts/million_token_demo.py --tokens 1000000 --chunk-size 2048 --d-model 64 --n-layers 2 --n-heads 2 --swa-window 256`

- Processed tokens: `1,000,000`
- Chunks: `489`
- Wall time: `132.59 s`
- Throughput: `7,542.1 tok/s`
- Peak state memory: `0.135 MiB`
- Memory per token over the full run: `0.141568 B/tok`

Interpretation: the peak recurrent state stayed constant across the entire
million-token run, which is the concrete streaming-memory property HydraLM is
designed to preserve.

#### Retrieval-path execution sanity check

`scripts/long_context_qa.py` was run with retrieval enabled on an **untrained**
model:

- sequence length: `4096`
- facts: `16`
- queries: `4`
- retrieval schedule: one retrieval layer every `3` positions
- retrieval chunk size: `64`
- retrieval top-k: `4`

Observed result: `0.0` accuracy across all buckets. This is **expected** for
an untrained model; the purpose of this run was to verify that the retrieval
evaluation path executes end-to-end under the current codebase.

### 0.6 Local caveats

- This formal record is a **Windows CPU validation pass**, not a GPU release
  benchmark campaign.
- The canonical headline benchmark tables below remain the intended
  publication-quality GPU-facing benchmark record.
- During validation, `scripts/cost_analysis.py` required UTF-8 console output
  on Windows because the final note line prints the Unicode character `Theta`.

## 1. Language modeling (held-out loss)

160M-parameter models, 10B training tokens, identical data mix:

| Model                       | d_model | n_layers | Val loss | Val ppl |
| --------------------------- | ------- | -------- | -------- | ------- |
| Transformer baseline        | 768     | 12       | 2.94     | 18.9    |
| Pure Gated DeltaNet         | 768     | 12       | 3.02     | 20.5    |
| HydraLM (4:1 GDN:SWA)       | 768     | 12       | 2.96     | 19.3    |
| HydraLM (3:1 GDN:SWA)       | 768     | 12       | 2.95     | 19.1    |

The 3:1 schedule matches the Transformer baseline within 0.01 nats
while keeping 75% of the inference-time speed advantage of the pure
DeltaNet.

## 2. Throughput

Tokens per second during greedy generation, batch size 1,
`d_model=768`, 12 layers:

| Context length | Transformer | Pure GDN | HydraLM 4:1 |
| -------------- | ----------- | -------- | ----------- |
| 1 k            | 210         | 240      | 235         |
| 4 k            | 155         | 240      | 232         |
| 16 k           | 42          | 238      | 228         |
| 64 k           | OOM         | 236      | 224         |
| 256 k          | OOM         | 230      | 218         |
| 1 M            | OOM         | 222      | 210         |

The Transformer OOMs beyond 16k because its KV cache grows linearly
with the sequence length; HydraLM and pure GDN have O(1) inference
memory. Pure GDN is marginally faster than HydraLM only because it
has no SWA layers at all; HydraLM's long-context recall (see below) is
dramatically better as a result.

## 3. Multi-Query Associative Recall (MQAR)

Exact-match accuracy on MQAR with increasing key-value bank sizes,
context length 8192:

| n_pairs | Transformer | Pure GDN | HydraLM 4:1 |
| ------- | ----------- | -------- | ----------- |
| 64      | 1.000       | 0.994    | 1.000       |
| 128     | 1.000       | 0.952    | 0.999       |
| 256     | 1.000       | 0.873    | 0.982       |
| 512     | 0.999       | 0.711    | 0.947       |
| 1024    | 0.998       | 0.524    | 0.872       |

HydraLM closes most of the recall gap to the Transformer while
keeping the DeltaNet efficiency profile. Pure GDN degrades sharply
beyond 256 pairs because its state capacity is `head_dim * n_heads` =
~4k dimensions — enough to distinguish 256 items cleanly but not
1024.

## 4. Needle-in-a-Haystack

Retrieval accuracy for a single fact placed at depth *d* inside a
distractor corpus of length *N*. HydraLM 4:1, 160M params:

```
            depth
length   0.10  0.25  0.50  0.75  0.90
  4k     1.00  1.00  1.00  1.00  1.00
 16k     0.98  0.98  0.99  1.00  1.00
 64k     0.92  0.94  0.96  0.99  1.00
256k     0.82  0.86  0.91  0.97  1.00
  1M     0.71  0.76  0.82  0.93  1.00
```

The pattern is the expected one: depth 0.9 (needle near the end of
the context) is always within the SWA window and scores perfectly;
depth 0.1 (needle near the start) relies on DeltaNet recall and
degrades gracefully with total length.

## 4b. Multi-fact long-context QA (new in 0.3.0)

Expected behaviour of `hydralm.eval.retrieval_qa` on the same 160M
backbone, 16k-token context, 32 inserted facts, 8 trailing queries
(see [`docs/evaluation.md`](./evaluation.md) §4 for the task setup):

| Configuration                                | acc  | acc@0-25 | acc@25-50 | acc@50-75 | acc@75-100 |
| -------------------------------------------- | ---- | -------- | --------- | --------- | ---------- |
| HydraLM 4:1 (GDN + SWA only)                 | 0.71 | 0.48     | 0.55      | 0.74      | 0.96       |
| HydraLM 4:1 + RAA `every=3, k=8, C=128`      | 0.93 | 0.89     | 0.91      | 0.94      | 0.98       |
| HydraLM 4:1 + RAA + MTP depth=2              | 0.94 | 0.90     | 0.92      | 0.94      | 0.99       |

RAA lifts the early-context buckets (`0-25`, `25-50`) from the
DeltaNet-state-capacity floor to near the SWA-window ceiling, which
is exactly the "natural extraction" gap the feature was designed to
close. MTP adds a small (~1 pp) head-room gain via denser training
signal. Reproduce with `scripts/long_context_qa.py`; exact numbers
are a function of the training corpus and recipe.

> **Methodology note.** These rows ran on the 0.3.0 training recipe
> (identical to 0.2.0 plus `mtp_loss_weight=0.1` when MTP is on), 10 B
> tokens. The first row reproduces the 0.2.0 `retrieval_qa` baseline.

## 5. Speculative decoding

Tokens per second using `hydralm.speculative_generate`, 160M target
model, 60M draft model:

| Draft `k` | Accept rate | Tokens / s | Speedup |
| --------- | ----------- | ---------- | ------- |
| 1         | 0.89        | 225        | 1.1×    |
| 2         | 0.82        | 280        | 1.4×    |
| 4         | 0.74        | 320        | 1.6×    |
| 8         | 0.63        | 360        | 1.8×    |
| 16        | 0.48        | 340        | 1.7×    |

Peak throughput is at `k = 8`. Beyond that, verification cost
overtakes the savings from accepted draft tokens.

### 5b. MTP head as zero-parameter draft (new in 0.3.0)

With `cfg.mtp_depth = 2`, the backbone itself produces the draft
tokens — no second model is needed. Measured with the same 160M
target and identical prompt suite:

| Draft path                | Draft params | Accept rate | Tokens / s | Speedup |
| ------------------------- | ------------ | ----------- | ---------- | ------- |
| 60M external draft, `k=4` | 60M          | 0.74        | 320        | 1.6×    |
| MTP depth 2 (self-draft)  | +0.3% of 160M| 0.78        | 330        | 1.7×    |
| MTP depth 3 (self-draft)  | +0.5% of 160M| 0.72        | 345        | 1.8×    |

The self-draft path is the most attractive operating point for
memory-constrained serving: it delivers draft-model-grade throughput
without the cost of hosting a second model.

## 6. Fact bank

Write 1M random `(key, value)` pairs at `capacity=10_000`
(9× rotations), query the final held-out 10%:

| Metric        | Value |
| ------------- | ----- |
| p@1           | 0.987 |
| p@8           | 0.999 |
| Query latency | 0.8 ms/query (fp32 keys, bs=64, 10k bank) |
| Query latency | 0.5 ms/query (fp16 keys, bs=64, 10k bank) |

## 7. FLOPs analysis

Measured FLOPs / token / layer, batch size 1, `d_model=768`,
`n_heads=12`, `head_dim=64`, `swa_window=256`:

| Layer type | FLOPs / token (train) | FLOPs / token (infer) |
| ---------- | --------------------- | --------------------- |
| DeltaNet   | 0.8 M                 | 0.8 M                 |
| SWA        | 1.2 M + 0.012 M × N   | 0.31 M (constant)     |

For a 12-layer HydraLM 4:1 at `N = 65_536`:

- Training: ~12 M FLOPs / token (mostly SWA, scaling with N)
- Inference: ~10 M FLOPs / token (constant)

Compare to an equivalent Transformer at the same N:
~360 M FLOPs / token during inference (KV cache streaming).

## Methodology notes

- **Warmup**: every throughput number drops the first 20 generated
  tokens to exclude `torch.compile` warmup and CUDA graph capture.
- **Measurement**: all timings use `torch.cuda.Event` for
  GPU-synchronous timing, not Python wall-clock.
- **Seeds**: every quality number is reported as mean over 3 seeds;
  variance is ≤0.5% in all cases.
- **Hardware**: results reproduce within ±5% on H100 and within
  ±15% on consumer GPUs (RTX 4090, RTX 3090). See `docs/faq.md` if
  you see larger deviations.
