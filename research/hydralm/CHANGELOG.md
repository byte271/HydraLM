# Changelog

All notable changes to HydraLM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Formal verification record in `docs/benchmarks.md` for the staged v0.3.0
  validation pass, including:
  - local Windows CPU environment details,
  - executed verification commands,
  - pytest results,
  - smoke-budget and paper-budget claim reproducer results,
  - CPU length-scaling sanity-check numbers,
  - million-token streaming demo results, and
  - retrieval-path execution notes.

### Changed
- Refreshed source-of-truth documentation to better match the currently
  exercised code paths and validation workflow:
  - `docs/deployment.md`
  - `docs/training.md`
  - `docs/faq.md`
  - `docs/glossary.md`
- Clarified in `docs/benchmarks.md` that the newly recorded local run is a
  Windows CPU validation pass and does not replace the canonical GPU-facing
  benchmark tables later in the document.

### Notes
- The validation record documents a local environment where:
  - 76 pytest cases passed,
  - the `HydraLMForCausalLM` save/load round-trip was manually reproduced
    inside the workspace after the fixture-backed pytest variant hit a
    Windows temporary-directory sandbox restriction, and
  - both smoke and paper claim reproducer budgets completed successfully.

### Planned
- Fused CUDA / Triton kernel for the gated delta-rule recurrence (currently
  the chunkwise kernel runs a tight PyTorch loop inside each chunk).
- Paged recurrent state in `deploy/compiled.py` so requests with different
  prompt lengths can share a single dynamic batch.
- Tensor-parallel support in `hydralm.training.Trainer` (FSDP is already
  wired; TP requires sharding the 2-D projections of DeltaNet / SWA / SwiGLU).
- INT8 / FP8 weight-only quantisation for the deploy path.
- Long-horizon `FactBank` compaction and TTL policies.

## [0.2.0] - 2026-04-21

Initial public research release. HydraLM is a sub-quadratic language model
that interleaves Gated DeltaNet mixers with a small fraction of
sliding-window attention layers (one SWA every four layers by default).

### Added

#### Core architecture (`hydralm`)

- `HydraConfig` dataclass with a computed `layer_types` schedule (tuple of
  `"deltanet"` / `"swa"`) derived from `n_layers` and `swa_every`.
- `HydraLM` module with a parallel `forward(input_ids, state=None, return_state=False)`
  for training / prefill and an `O(1)` `step(input_ids, state)` for decoding.
- `HydraBlock`: pre-norm residual unit dispatching to `GatedDeltaNet` or
  `SlidingWindowAttention` based on `cfg.layer_types[i]`.
- `GatedDeltaNet` mixer: per-token forget gate `alpha` and write strength
  `beta`, L2-normalised queries/keys, Mamba-style short causal convolution,
  optional SiLU output gate.
- `SlidingWindowAttention` mixer: causal attention with bounded window
  (default 512), rotary positional embedding, explicit `k_cache`/`v_cache`
  ring buffer for mid-stream continuation.
- Support modules: `RMSNorm`, `SwiGLU`, `RotaryEmbedding`, `ShortConv`.

#### Kernels (`hydralm.kernels.delta_rule`)

- `delta_rule_reference`: slow, auditable scalar reference.
- `delta_rule_recurrent`: allocation-free torch-scripted hot loop used for
  streaming inference.
- `delta_rule_chunkwise`: the form used for training — chunkwise intra-chunk
  `(C x C)` matmul plus a single `(Dv x Dk)` inter-chunk update per chunk.

#### Inference

- `hydralm.generate`: batched sampling with temperature, top-k, and top-p,
  driven by the unified recurrent state.
- `hydralm.streaming.StreamingEngine`: chunked prefill and `extend_and_generate`
  with constant memory in the sequence length.
- `hydralm.speculative_generate` + `SpecDecodingStats`: exact
  draft/target speculative decoding that correctly rolls back HydraLM's
  recurrent state on rejection by cloning per-round snapshots.

#### Memory

- `hydralm.memory.FactBank`: a zero-gradient associative memory that writes
  `(key, value)` pairs into the same delta-rule recurrence used by the
  linear layers, with cosine/MSE/argmax retrieval scores.

#### Optimiser (`hydralm.optim`)

- `Muon` — Newton–Schulz-orthogonalised momentum optimiser for 2-D weight
  matrices, with Nesterov look-ahead and aspect-ratio LR correction.
- `HybridMuonAdamW`: combined optimiser routing 2-D hidden-layer matrices
  to Muon and embeddings / LM head / biases / norms to AdamW.
- `build_hybrid_optimizer(model, ...)`: automatic parameter split.
- `zeropower_via_newton_schulz`: the underlying orthogonalisation primitive.

#### Training (`hydralm.training`)

- `TrainingConfig` dataclass with cosine-with-warmup schedule, gradient
  accumulation, gradient clipping, AMP (bf16/fp16/auto), FSDP auto-wrap
  at `HydraBlock` granularity, optional `torch.compile`, optional gradient
  checkpointing per block, and optimiser selection (`"adamw"` or
  `"muon"`).
- `Trainer.fit(data, on_eval=None)` — minimal audit-friendly loop.

#### Evaluation (`hydralm.eval`)

- `mqar`: multi-query associative recall (Arora et al., 2024).
- `long_context`: needle-in-a-haystack generator and scorer.
- `online_learning`: capacity, interference, overwrite, and KV-cache memory
  comparisons for `FactBank`.
- `claims`: six numerical gates that define the HydraLM contract
  (linear complexity, lossless MQAR, constant state, cost reduction,
  drop-in replacement, zero-gradient online learning), consumed by both
  `tests/test_claims.py` and `scripts/reproduce_claims.py`.

#### Deployment (`hydralm.deploy`)

- `HydraLMForCausalLM`: `transformers`-style causal-LM adapter with
  `forward`, `generate`, `save_pretrained`, and `from_pretrained`.
- `CompiledDecoder` + `Request`: batched low-latency decoder backed by
  `torch.compile(mode="reduce-overhead")` over `HydraLM.step`.

#### Baselines (`hydralm.baselines`)

- `DenseTransformer`: reference dense Transformer implementing the same
  interface as `HydraLM` for like-for-like comparisons.
- `flops`: closed-form FLOPs / memory / dollar model shared between the
  claim gates and `scripts/cost_analysis.py`.

#### Reproduction scripts

- `train_tiny.py`, `benchmark_length.py`, `cost_analysis.py`,
  `million_token_demo.py`, `needle_in_haystack.py`,
  `online_learning_demo.py`, `reproduce_claims.py`, `run_mqar.py`.

#### Tests

- Shape correctness; recurrent ↔ parallel equivalence of the delta-rule
  kernels; streaming-chunk state bit-identity; speculative-decoding
  correctness; `FactBank` dynamics (memorise, overwrite, capacity);
  Muon orthogonalisation; complexity scaling; and the full claim harness.

### Fixed
- `HydraLMForCausalLM.from_pretrained` now restores a persisted
  `layer_types` schedule instead of dropping it and recomputing a default.
- `HydraConfig.to_dict` no longer references a non-existent `betas`
  attribute left over from an earlier iteration.

[Unreleased]: https://github.com/byte271/hydralm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/byte271/hydralm/releases/tag/v0.2.0
