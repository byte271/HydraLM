# HydraLM Monorepo

This repository hosts **HydraLM** — a sub-quadratic language model that combines
**Gated DeltaNet** linear attention with sparse **Sliding-Window Attention**, plus
the surrounding tooling (training scripts, evaluation harnesses, deployment
adapters, and reproducible-claims infrastructure).

The research codebase lives under [`research/hydralm/`](./research/hydralm/).
Everything listed below is a direct link into that tree so this file can be
used as a single navigational entry point.

---

## Table of contents

- [Project metadata](#project-metadata)
- [Top-level project documents](#top-level-project-documents)
- [Documentation (`research/hydralm/docs/`)](#documentation-researchhydralmdocs)
- [Results & reproducibility](#results--reproducibility)
- [Source layout (`research/hydralm/hydralm/`)](#source-layout-researchhydralmhydralm)
  - [Core model](#core-model)
  - [Modules (`modules/`)](#modules-modules)
  - [Kernels (`kernels/`)](#kernels-kernels)
  - [Baselines (`baselines/`)](#baselines-baselines)
  - [Optimizers (`optim/`)](#optimizers-optim)
  - [Training (`training/`)](#training-training)
  - [Evaluation (`eval/`)](#evaluation-eval)
  - [Memory (`memory/`)](#memory-memory)
  - [Deployment (`deploy/`)](#deployment-deploy)
- [Scripts (`research/hydralm/scripts/`)](#scripts-researchhydralmscripts)
- [Tests (`research/hydralm/tests/`)](#tests-researchhydralmtests)
- [Data (`research/hydralm/data/`)](#data-researchhydralmdata)
- [Build & packaging](#build--packaging)
- [Quick start](#quick-start)
- [License & citation](#license--citation)

---

## Project metadata

- **Package name:** `hydralm`
- **Version:** `0.2.0`
- **Python:** `>= 3.10`
- **License:** MIT — see [`research/hydralm/LICENSE`](./research/hydralm/LICENSE)
- **Build / dependency manifest:** [`research/hydralm/pyproject.toml`](./research/hydralm/pyproject.toml)
- **Resolved lockfile:** [`research/hydralm/uv.lock`](./research/hydralm/uv.lock)
- **Citation file:** [`research/hydralm/CITATION.cff`](./research/hydralm/CITATION.cff)

Generated packaging metadata (written by `setuptools`) is kept under
[`research/hydralm/hydralm.egg-info/`](./research/hydralm/hydralm.egg-info/):

- [`PKG-INFO`](./research/hydralm/hydralm.egg-info/PKG-INFO)
- [`SOURCES.txt`](./research/hydralm/hydralm.egg-info/SOURCES.txt)
- [`requires.txt`](./research/hydralm/hydralm.egg-info/requires.txt)
- [`top_level.txt`](./research/hydralm/hydralm.egg-info/top_level.txt)
- [`dependency_links.txt`](./research/hydralm/hydralm.egg-info/dependency_links.txt)

---

## Top-level project documents

These files live at the root of `research/hydralm/` and describe the project,
its history, and its community contract.

| Document | Path | Purpose |
| --- | --- | --- |
| Authors | [`research/hydralm/AUTHORS.md`](./research/hydralm/AUTHORS.md) | People who wrote the code. |
| Changelog | [`research/hydralm/CHANGELOG.md`](./research/hydralm/CHANGELOG.md) | Version-by-version history of what changed. |
| Citation | [`research/hydralm/CITATION.cff`](./research/hydralm/CITATION.cff) | Machine-readable citation metadata (CFF 1.2). |
| Code of Conduct | [`research/hydralm/CODE_OF_CONDUCT.md`](./research/hydralm/CODE_OF_CONDUCT.md) | Community behavior expectations. |
| Contributing | [`research/hydralm/CONTRIBUTING.md`](./research/hydralm/CONTRIBUTING.md) | How to submit changes, run the test suite, and open PRs. |
| License | [`research/hydralm/LICENSE`](./research/hydralm/LICENSE) | MIT license text. |
| Results | [`research/hydralm/RESULTS.md`](./research/hydralm/RESULTS.md) | Auto-generated pass/fail report for every numerical claim. |
| Security | [`research/hydralm/SECURITY.md`](./research/hydralm/SECURITY.md) | Supported versions and how to report vulnerabilities. |

---

## Documentation (`research/hydralm/docs/`)

Long-form documentation, organized by topic. Each file is self-contained and
is linked from the others where relevant.

| Topic | Path | What's inside |
| --- | --- | --- |
| Architecture | [`research/hydralm/docs/architecture.md`](./research/hydralm/docs/architecture.md) | Block diagram, layer scheduling, hybrid DeltaNet + SWA design rationale. |
| API reference | [`research/hydralm/docs/api.md`](./research/hydralm/docs/api.md) | Public Python API: `HydraConfig`, `HydraLM`, generation helpers. |
| Benchmarks | [`research/hydralm/docs/benchmarks.md`](./research/hydralm/docs/benchmarks.md) | Throughput, memory, and quality numbers across context lengths. |
| Claims | [`research/hydralm/docs/claims.md`](./research/hydralm/docs/claims.md) | Enumerated quantitative claims (C1–C9) with pass thresholds. |
| Deployment | [`research/hydralm/docs/deployment.md`](./research/hydralm/docs/deployment.md) | `torch.compile`, HuggingFace adapter, serving patterns. |
| Evaluation | [`research/hydralm/docs/evaluation.md`](./research/hydralm/docs/evaluation.md) | MQAR, Needle-in-a-Haystack, long-context, online-learning protocols. |
| Fact bank | [`research/hydralm/docs/fact-bank.md`](./research/hydralm/docs/fact-bank.md) | Design of the external associative memory module. |
| FAQ | [`research/hydralm/docs/faq.md`](./research/hydralm/docs/faq.md) | Frequently asked questions. |
| Glossary | [`research/hydralm/docs/glossary.md`](./research/hydralm/docs/glossary.md) | Terminology reference (GDN, SWA, RoPE, RMSNorm, …). |
| Roadmap | [`research/hydralm/docs/roadmap.md`](./research/hydralm/docs/roadmap.md) | What's planned next and what is explicitly out of scope. |
| Theory | [`research/hydralm/docs/theory.md`](./research/hydralm/docs/theory.md) | Delta-rule derivation and recall/efficiency tradeoff analysis. |
| Training | [`research/hydralm/docs/training.md`](./research/hydralm/docs/training.md) | Data pipeline, optimizer schedule, hybrid Muon+AdamW recipe. |

---

## Results & reproducibility

HydraLM's numerical claims are mechanically checked by CI. Every claim has a
script, a pass threshold, and a test.

- **Human-readable summary** — [`research/hydralm/RESULTS.md`](./research/hydralm/RESULTS.md)
- **Raw measurements (JSON)** — [`research/hydralm/results.json`](./research/hydralm/results.json)
- **Claim definitions** — [`research/hydralm/docs/claims.md`](./research/hydralm/docs/claims.md)
- **Runner** — [`research/hydralm/scripts/reproduce_claims.py`](./research/hydralm/scripts/reproduce_claims.py)

To regenerate the results file:

```bash
cd research/hydralm
python scripts/reproduce_claims.py --budget paper --out RESULTS.md
```

---

## Source layout (`research/hydralm/hydralm/`)

The installable Python package. Top-level entry point:
[`research/hydralm/hydralm/__init__.py`](./research/hydralm/hydralm/__init__.py).

### Core model

| File | Role |
| --- | --- |
| [`hydralm/config.py`](./research/hydralm/hydralm/config.py) | `HydraConfig` dataclass — all model hyperparameters. |
| [`hydralm/model.py`](./research/hydralm/hydralm/model.py) | `HydraLM` top-level `nn.Module`: embeddings, block stack, tied LM head. |
| [`hydralm/generation.py`](./research/hydralm/hydralm/generation.py) | Non-streaming greedy / sampling generation loop. |
| [`hydralm/streaming.py`](./research/hydralm/hydralm/streaming.py) | Step-wise generation that threads the recurrent state across calls. |
| [`hydralm/spec_decoding.py`](./research/hydralm/hydralm/spec_decoding.py) | Speculative decoding with a draft model. |
| [`hydralm/utils.py`](./research/hydralm/hydralm/utils.py) | Shared helpers (seeding, parameter counts, pretty-printing). |

### Modules (`modules/`)

Reusable `nn.Module` building blocks. Package init:
[`hydralm/modules/__init__.py`](./research/hydralm/hydralm/modules/__init__.py).

| File | Role |
| --- | --- |
| [`modules/block.py`](./research/hydralm/hydralm/modules/block.py) | Pre-norm residual block (mixer + SwiGLU MLP). |
| [`modules/gated_deltanet.py`](./research/hydralm/hydralm/modules/gated_deltanet.py) | Gated DeltaNet linear-attention layer. |
| [`modules/sliding_window.py`](./research/hydralm/hydralm/modules/sliding_window.py) | Sliding-Window Attention with rolling KV cache. |
| [`modules/swiglu.py`](./research/hydralm/hydralm/modules/swiglu.py) | SwiGLU feed-forward MLP. |
| [`modules/rmsnorm.py`](./research/hydralm/hydralm/modules/rmsnorm.py) | RMSNorm layer. |
| [`modules/rotary.py`](./research/hydralm/hydralm/modules/rotary.py) | Rotary position embeddings (RoPE). |
| [`modules/short_conv.py`](./research/hydralm/hydralm/modules/short_conv.py) | Causal depthwise 1-D convolution used before GDN. |

### Kernels (`kernels/`)

Low-level numerical primitives. Package init:
[`hydralm/kernels/__init__.py`](./research/hydralm/hydralm/kernels/__init__.py).

| File | Role |
| --- | --- |
| [`kernels/delta_rule.py`](./research/hydralm/hydralm/kernels/delta_rule.py) | Gated delta-rule recurrence — parallel (training) and step-wise (inference) forms. |

### Baselines (`baselines/`)

Reference implementations used for head-to-head comparison. Package init:
[`hydralm/baselines/__init__.py`](./research/hydralm/hydralm/baselines/__init__.py).

| File | Role |
| --- | --- |
| [`baselines/transformer.py`](./research/hydralm/hydralm/baselines/transformer.py) | Plain softmax-attention Transformer of matched parameter count. |
| [`baselines/flops.py`](./research/hydralm/hydralm/baselines/flops.py) | Analytic FLOP / memory estimators for HydraLM and the baseline. |

### Optimizers (`optim/`)

Training-time optimizers. Package init:
[`hydralm/optim/__init__.py`](./research/hydralm/hydralm/optim/__init__.py).

| File | Role |
| --- | --- |
| [`optim/muon.py`](./research/hydralm/hydralm/optim/muon.py) | `HybridMuonAdamW` — Muon for matrix-shaped params, AdamW for the rest. |

### Training (`training/`)

Training loop and related utilities. Package init:
[`hydralm/training/__init__.py`](./research/hydralm/hydralm/training/__init__.py).

| File | Role |
| --- | --- |
| [`training/trainer.py`](./research/hydralm/hydralm/training/trainer.py) | Minimal trainer with gradient accumulation, LR scheduling, and logging. |

### Evaluation (`eval/`)

Evaluation harnesses that back the numerical claims. Package init:
[`hydralm/eval/__init__.py`](./research/hydralm/hydralm/eval/__init__.py).

| File | Role |
| --- | --- |
| [`eval/claims.py`](./research/hydralm/hydralm/eval/claims.py) | Claim registry — the machine-readable list of what must hold. |
| [`eval/long_context.py`](./research/hydralm/hydralm/eval/long_context.py) | Long-context throughput / memory evaluation. |
| [`eval/mqar.py`](./research/hydralm/hydralm/eval/mqar.py) | Multi-Query Associative Recall benchmark. |
| [`eval/online_learning.py`](./research/hydralm/hydralm/eval/online_learning.py) | Online / test-time fact-bank evaluation. |

### Memory (`memory/`)

External associative-memory module. Package init:
[`hydralm/memory/__init__.py`](./research/hydralm/hydralm/memory/__init__.py).

| File | Role |
| --- | --- |
| [`memory/fact_bank.py`](./research/hydralm/hydralm/memory/fact_bank.py) | `FactBank` — write/query key-value store for test-time learning. |

### Deployment (`deploy/`)

Adapters that expose HydraLM through external APIs. Package init:
[`hydralm/deploy/__init__.py`](./research/hydralm/hydralm/deploy/__init__.py).

| File | Role |
| --- | --- |
| [`deploy/compiled.py`](./research/hydralm/hydralm/deploy/compiled.py) | `torch.compile` wrapper for production inference. |
| [`deploy/hf_adapter.py`](./research/hydralm/hydralm/deploy/hf_adapter.py) | HuggingFace `PreTrainedModel` / `generate()` shim. |

---

## Scripts (`research/hydralm/scripts/`)

Runnable entry points. Each script is documented in-file via `--help`.

| Script | Path | Purpose |
| --- | --- | --- |
| Benchmark context length | [`scripts/benchmark_length.py`](./research/hydralm/scripts/benchmark_length.py) | Measures per-token latency across context lengths (backs **C5**). |
| Cost analysis | [`scripts/cost_analysis.py`](./research/hydralm/scripts/cost_analysis.py) | FLOP/memory/$ savings vs. a matched Transformer. |
| Million-token demo | [`scripts/million_token_demo.py`](./research/hydralm/scripts/million_token_demo.py) | Streams 1M+ tokens with constant memory (backs **C4**). |
| Needle-in-a-Haystack | [`scripts/needle_in_haystack.py`](./research/hydralm/scripts/needle_in_haystack.py) | Long-context retrieval benchmark (backs **C8**). |
| Online learning demo | [`scripts/online_learning_demo.py`](./research/hydralm/scripts/online_learning_demo.py) | Fact-bank writes + queries, no gradients (backs **C9**). |
| Reproduce claims | [`scripts/reproduce_claims.py`](./research/hydralm/scripts/reproduce_claims.py) | Runs every claim and writes `RESULTS.md` + `results.json`. |
| MQAR | [`scripts/run_mqar.py`](./research/hydralm/scripts/run_mqar.py) | Multi-Query Associative Recall runner (backs **C7**). |
| Train tiny | [`scripts/train_tiny.py`](./research/hydralm/scripts/train_tiny.py) | End-to-end training on TinyShakespeare for smoke tests. |

---

## Tests (`research/hydralm/tests/`)

Pytest suite. Run with `pytest` from `research/hydralm/`.

| Test file | Path | Covers |
| --- | --- | --- |
| Claims | [`tests/test_claims.py`](./research/hydralm/tests/test_claims.py) | Registry integrity and claim-runner wiring. |
| Complexity | [`tests/test_complexity.py`](./research/hydralm/tests/test_complexity.py) | O(N) scaling slope — backs **C4**. |
| Equivalence | [`tests/test_equivalence.py`](./research/hydralm/tests/test_equivalence.py) | Parallel vs. streaming logit equality — backs **C1**. |
| Eval & adapter | [`tests/test_eval_and_adapter.py`](./research/hydralm/tests/test_eval_and_adapter.py) | Eval harnesses and HuggingFace adapter. |
| Fact bank | [`tests/test_fact_bank.py`](./research/hydralm/tests/test_fact_bank.py) | Round-trip and rotation behavior — backs **C3**. |
| Muon | [`tests/test_muon.py`](./research/hydralm/tests/test_muon.py) | Optimizer math and wall-clock advantage — backs **C6**. |
| Shapes | [`tests/test_shapes.py`](./research/hydralm/tests/test_shapes.py) | Tensor-shape contracts and batch independence — backs **C2**. |
| Speculative decoding | [`tests/test_spec_decoding.py`](./research/hydralm/tests/test_spec_decoding.py) | Draft-model accept/reject correctness. |
| Streaming | [`tests/test_streaming.py`](./research/hydralm/tests/test_streaming.py) | Stateful generation correctness. |

---

## Data (`research/hydralm/data/`)

| File | Path | Notes |
| --- | --- | --- |
| TinyShakespeare | [`data/tinyshakespeare.txt`](./research/hydralm/data/tinyshakespeare.txt) | Public-domain corpus used by `scripts/train_tiny.py` for smoke tests. |

---

## Build & packaging

- **Build manifest:** [`research/hydralm/pyproject.toml`](./research/hydralm/pyproject.toml)
  - Declares runtime deps (`torch`, `einops`, `numpy`) and optional extras
    (`dev`, `train`, `hf`).
  - Configures `pytest` and `ruff`.
- **Lockfile:** [`research/hydralm/uv.lock`](./research/hydralm/uv.lock)
- **Generated egg-info:** [`research/hydralm/hydralm.egg-info/`](./research/hydralm/hydralm.egg-info/) — `PKG-INFO`, `SOURCES.txt`, `requires.txt`, `top_level.txt`, `dependency_links.txt`.

---

## Quick start

```bash
cd research/hydralm

# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the test suite
pytest

# Reproduce every published claim
python scripts/reproduce_claims.py --budget paper --out RESULTS.md

# Smoke-train on TinyShakespeare
python scripts/train_tiny.py
```

See [`research/hydralm/docs/training.md`](./research/hydralm/docs/training.md)
for a full training recipe and
[`research/hydralm/docs/deployment.md`](./research/hydralm/docs/deployment.md)
for serving guidance.

---

## License & citation

HydraLM is released under the MIT License
([`research/hydralm/LICENSE`](./research/hydralm/LICENSE)).

If you use HydraLM in academic work, please cite it using the metadata in
[`research/hydralm/CITATION.cff`](./research/hydralm/CITATION.cff).
