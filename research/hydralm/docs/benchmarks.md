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
