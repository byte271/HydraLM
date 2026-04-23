# HydraLM

HydraLM is a hybrid sub-quadratic language model that combines three
complementary memory paths:

- **Gated DeltaNet** for fast recurrent sequence modeling
- **Sliding-Window Attention** for exact local recall
- **Retrieval Attention** for chunk-sparse long-range information access

The result is a model family designed to keep the scaling profile of
linear-time backbones while recovering the kinds of exact recall that
long-context language modeling usually needs.

Website: [hydralm.pages.dev](https://hydralm.pages.dev)  
GitHub: [byte271/HydraLM](https://github.com/byte271/HydraLM)

## Why this repo exists

This repository packages the full HydraLM project in one place:

- the installable Python package under [`research/hydralm/`](./research/hydralm/)
- the long-form technical docs under [`research/hydralm/docs/`](./research/hydralm/docs/)
- the static website in [`website/`](./website/)
- the reproducibility and claims infrastructure used to ground published behavior

If you want the fastest entry point, start with the package README at
[`research/hydralm/README.md`](./research/hydralm/README.md).

## What shipped in 0.3.0

Version `0.3.0` adds three opt-in capabilities aimed at natural long-range
information extraction:

- **Retrieval Attention**  
  Chunk-sparse top-k routing over prior chunks, letting the model recover
  relevant distant context without paying full quadratic attention cost.

- **Compressive Memory**  
  A three-tier KV memory wrapper that keeps serving memory bounded while the
  effective context continues to grow.

- **Multi-Token Prediction (MTP)**  
  A next-k auxiliary head that densifies the training signal and doubles as a
  self-drafting path for speculative decoding.

All three are opt-in. The default `HydraConfig()` preserves the legacy 0.2.0
behavior until you turn them on.

## Quick start

```bash
git clone https://github.com/byte271/HydraLM.git
cd HydraLM/research/hydralm

# Install with development dependencies
pip install -e ".[dev]"

# Run the test suite
pytest

# Reproduce the claims report
python scripts/reproduce_claims.py --budget paper --out RESULTS.md
```

## Where to look first

| If you want to... | Start here |
| --- | --- |
| Understand the package and workflows | [`research/hydralm/README.md`](./research/hydralm/README.md) |
| Understand the hybrid model design | [`research/hydralm/docs/architecture.md`](./research/hydralm/docs/architecture.md) |
| Dive into retrieval, memory, and MTP | [`research/hydralm/docs/retrieval.md`](./research/hydralm/docs/retrieval.md) |
| Review benchmarks and formal claims | [`research/hydralm/docs/claims.md`](./research/hydralm/docs/claims.md) |
| Inspect deployment guidance | [`research/hydralm/docs/deployment.md`](./research/hydralm/docs/deployment.md) |

## Repository layout

```text
HydraLM/
├── README.md
├── website/
│   ├── index.html
│   └── docs.html
└── research/
    └── hydralm/
        ├── README.md
        ├── CHANGELOG.md
        ├── docs/
        ├── hydralm/
        ├── scripts/
        ├── tests/
        └── data/
```

## Project highlights

- **Sub-quadratic training profile** driven by DeltaNet-heavy hybrid scheduling
- **Constant-memory streaming path** for long-form inference
- **Claims-backed development** with reproducible scripts and result artifacts
- **Pure PyTorch implementation** with no custom CUDA build required for the base path
- **Documentation-first structure** with package docs, theory notes, evaluation guides, and a static site

## Reproducibility and verification

HydraLM includes a reproducibility lane for the main published behaviors:

- Human-readable results: [`research/hydralm/RESULTS.md`](./research/hydralm/RESULTS.md)
- Structured measurements: [`research/hydralm/results.json`](./research/hydralm/results.json)
- Claim definitions: [`research/hydralm/docs/claims.md`](./research/hydralm/docs/claims.md)
- Claim runner: [`research/hydralm/scripts/reproduce_claims.py`](./research/hydralm/scripts/reproduce_claims.py)

The package README and the docs directory go deeper into individual harnesses
such as MQAR, needle-in-a-haystack, long-context QA, and streaming validation.

## Core entry points

Some of the most important source files:

- [`research/hydralm/hydralm/config.py`](./research/hydralm/hydralm/config.py)
- [`research/hydralm/hydralm/model.py`](./research/hydralm/hydralm/model.py)
- [`research/hydralm/hydralm/modules/retrieval_attention.py`](./research/hydralm/hydralm/modules/retrieval_attention.py)
- [`research/hydralm/hydralm/modules/compressive_memory.py`](./research/hydralm/hydralm/modules/compressive_memory.py)
- [`research/hydralm/hydralm/modules/mtp_head.py`](./research/hydralm/hydralm/modules/mtp_head.py)
- [`research/hydralm/hydralm/streaming.py`](./research/hydralm/hydralm/streaming.py)

## Documentation map

The long-form docs are organized by topic:

- Architecture: [`research/hydralm/docs/architecture.md`](./research/hydralm/docs/architecture.md)
- API reference: [`research/hydralm/docs/api.md`](./research/hydralm/docs/api.md)
- Benchmarks: [`research/hydralm/docs/benchmarks.md`](./research/hydralm/docs/benchmarks.md)
- Deployment: [`research/hydralm/docs/deployment.md`](./research/hydralm/docs/deployment.md)
- Evaluation: [`research/hydralm/docs/evaluation.md`](./research/hydralm/docs/evaluation.md)
- FAQ: [`research/hydralm/docs/faq.md`](./research/hydralm/docs/faq.md)
- Glossary: [`research/hydralm/docs/glossary.md`](./research/hydralm/docs/glossary.md)
- Retrieval and long-range features: [`research/hydralm/docs/retrieval.md`](./research/hydralm/docs/retrieval.md)
- Roadmap: [`research/hydralm/docs/roadmap.md`](./research/hydralm/docs/roadmap.md)
- Theory: [`research/hydralm/docs/theory.md`](./research/hydralm/docs/theory.md)
- Training: [`research/hydralm/docs/training.md`](./research/hydralm/docs/training.md)


## License and citation

HydraLM is released under the MIT License:
[`research/hydralm/LICENSE`](./research/hydralm/LICENSE)

If you use HydraLM in academic or research work, cite it via:
[`research/hydralm/CITATION.cff`](./research/hydralm/CITATION.cff)
