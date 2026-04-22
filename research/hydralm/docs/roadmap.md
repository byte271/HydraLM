# Roadmap

This roadmap lists the features and research directions on the
HydraLM team's table. Items are grouped by confidence, not by date —
we deliberately avoid calendar-bound promises because HydraLM is a
research reference implementation, not a shipped product.

Upstream tracking for every item below lives at
`https://github.com/byte271/hydralm/issues`.

## Confirmed (in progress or queued)

### Flash-Attention 3 backend for SWA

The current SWA layer is a plain PyTorch implementation. An FA3
kernel would cut per-token SWA latency by ~2× and make the long-
context benchmark numbers in `docs/benchmarks.md` improve
proportionally.

- Tracking: `byte271/hydralm#12`
- Acceptance criteria: `tests/test_equivalence.py` passes with the
  FA3 path enabled, and `scripts/benchmark_length.py` shows ≥1.5×
  speedup at `N ≥ 64k`.

### Multi-GPU FSDP reference

The current `Trainer` supports DDP on a single node. An FSDP path
would let researchers train ~1B-parameter HydraLMs on 8× A100
nodes without rewriting the trainer.

- Tracking: `byte271/hydralm#18`
- Acceptance criteria: a dedicated `scripts/train_fsdp.py` that
  reproduces the 160M numbers on 4 GPUs at matched loss, same
  wall-clock per token.

### Fact-bank IVF / HNSW backends

Current fact-bank retrieval is exact. For banks above ~1M entries,
an approximate-nearest-neighbor backend becomes necessary. We plan
two adapters: FAISS (IVF-Flat) and a native HNSW implementation for
systems without FAISS.

- Tracking: `byte271/hydralm#21`
- Acceptance criteria: query latency under 2 ms at 100M entries on
  a single A100; p@1 within 2 percentage points of exact search.

## Likely (design sketch exists, not yet implemented)

### Grouped-query attention for SWA

Replacing multi-head SWA with grouped-query attention halves the KV
cache inside the SWA window at roughly matched quality, based on the
Llama 3 and Qwen 2 ablations.

### Mixture-of-experts MLPs

A natural axis of scaling for the SwiGLU MLP. We'd expose
`cfg.n_experts` and `cfg.n_active_experts` so that downstream users
can toggle between dense and sparse MLPs without rewriting the
block.

### In-loop claim regressions

Every training run would, at predefined step intervals, run the claim
benchmarks from `docs/claims.md` on the current checkpoint and log
them to W&B / TensorBoard. Cheap enough that we can enable it by
default.

### Training recipe for 1B parameters

The 160M recipe in `docs/training.md` is battle-tested. Scaling it to
1B is "just" a matter of running the numbers, but we want to publish
a reproducible recipe with documented hardware cost and final
numbers before declaring it a supported scale.

## Exploratory (research questions we find interesting)

### Learned layer-type schedules

`cfg.layer_types` is currently a user-specified list. A small
learned controller that chooses `deltanet` / `swa` per layer during
training — evaluated via differentiable search — could find
better-than-default schedules at matched parameter count.

### State-transfer between DeltaNet and SWA

DeltaNet's state matrix and SWA's KV cache encode different views of
the past. A learned projection that initializes an SWA layer's KV
cache from the previous DeltaNet state could make the hybrid more
sample-efficient.

### Compressive fact banks

A fact bank that periodically *compresses* older entries (e.g. via
k-means or product quantization) would trade a small precision loss
for much higher effective capacity. Promising for agent memory use
cases where older facts matter less.

### Parallelized delta-rule training kernel

The current delta-rule kernel is scan-parallel (O(log N) depth,
O(N) work per layer). A fully chunk-parallel implementation along
the lines of the Parallel DeltaNet paper would reduce training
wall-clock further at the cost of additional kernel complexity.

## Explicitly out of scope

The following are common asks that we do not intend to implement in
this repository. They are valuable and welcome in downstream forks;
they simply are not part of HydraLM's mission of being a *clean
reference* implementation.

- **Custom tokenizer training.** Use `tokenizers` or `sentencepiece`
  directly.
- **Instruction-tuning recipes.** Use `axolotl`, `openrlhf`, `trl`,
  or similar.
- **Production serving stack.** Export to HF and use `vllm` / `TGI` /
  `sglang`.
- **Web UI / chat interface.** HydraLM is a layer library; chat UI
  belongs in a separate repo.

## How to influence the roadmap

1. Open an issue at
   `https://github.com/byte271/hydralm/issues` describing the
   problem you want solved, not a specific solution.
2. If you want to implement something listed above, please comment
   on the tracking issue before starting work so we can help you
   scope it against the acceptance criteria.
3. For research-flavored proposals, a short design doc as the issue
   body is always appreciated. Prior art and expected benchmark
   impact are the two most useful sections.

See `CONTRIBUTING.md` for the mechanical workflow.
