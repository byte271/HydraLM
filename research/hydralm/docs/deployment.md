# Deployment Guide

This document covers inference-time deployment of HydraLM: loading a
trained checkpoint, running single-token and streaming generation,
exporting to Hugging Face format, and compiling a latency-optimized
graph with `torch.compile`.

## Loading a checkpoint

```python
from hydralm import HydraLM

model = HydraLM.from_pretrained("checkpoints/step-50000").cuda().eval()
```

`from_pretrained` reads `config.json` + `pytorch_model.bin` (or
`model.safetensors`) from the given directory or Hugging Face Hub
repo id (e.g. `"byte271/hydralm-160m"`).

## Single-shot generation

```python
import torch
from hydralm import HydraLM, generate

model = HydraLM.from_pretrained("byte271/hydralm-160m").cuda().eval()

input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
out = generate(
    model, input_ids,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
)
# out: (1, 4 + 128) tensor of token ids
```

`generate` internally uses the incremental decoding path, so memory
and FLOPs per token are O(state) — independent of the total number of
tokens produced.

## Streaming generation

For interactive applications, use `hydralm.streaming.stream` to
receive one token at a time:

```python
from hydralm.streaming import stream

for tok in stream(model, input_ids, max_new_tokens=256, temperature=0.8):
    print(tokenizer.decode([tok]), end="", flush=True)
```

The generator yields integers; wrap it with your tokenizer of choice.

## Speculative decoding

Speculative decoding accelerates generation by having a small **draft
model** propose `k` tokens, which the larger **target model** then
verifies in a single parallel forward pass. HydraLM ships a reference
implementation in `hydralm.spec_decoding`:

```python
from hydralm import speculative_generate, HydraLM

draft  = HydraLM.from_pretrained("byte271/hydralm-60m").cuda().eval()
target = HydraLM.from_pretrained("byte271/hydralm-160m").cuda().eval()

out = speculative_generate(
    draft, target, input_ids,
    max_new_tokens=256,
    draft_k=4,
)
```

Expected speedup on matched hardware:

| Setup                               | Tokens / s | Speedup |
| ----------------------------------- | ---------- | ------- |
| Target only (160M)                  | 180        | 1.0×    |
| Draft (60M) + Target (160M), `k=4`  | 320        | 1.8×    |
| Draft (60M) + Target (160M), `k=8`  | 360        | 2.0×    |

`k` beyond 8 tends to hit diminishing returns; see
`docs/benchmarks.md`.

## Compiled inference graph

For minimum-latency deployment, compile the step function once:

```python
from hydralm.deploy.compiled import compile_for_inference

step_fn = compile_for_inference(model, batch_size=1, max_length=4096)

# Reuse `step_fn` across many requests. First call warms up the graph.
out = step_fn(input_ids, state=None)
```

`compile_for_inference` is a thin wrapper around `torch.compile` with
mode `"reduce-overhead"` and CUDA graphs enabled. It typically shaves
15–25% off per-token latency on A100 for sequence lengths up to
~64k.

Because CUDA graphs require static shapes, the compiled `step_fn`
accepts one token per call. If you need batched inference with
different sequence lengths, use the eager path — the speedup of CUDA
graphs does not survive dynamic shapes.

## Hugging Face transformers export

Every HydraLM checkpoint can be loaded by `transformers.AutoModel`
through a thin adapter:

```python
from hydralm.deploy.hf_adapter import export_to_hf

export_to_hf(
    checkpoint="checkpoints/step-50000",
    output_dir="hf/hydralm-160m",
)

# Afterwards:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "hf/hydralm-160m", trust_remote_code=True,
)
```

The exported directory contains `config.json`, a `model.safetensors`
shard, and a small `modeling_hydralm.py` file that registers the
model with the `transformers` auto-classes. `trust_remote_code=True`
is required because HydraLM defines a custom architecture not present
in upstream `transformers`.

## Quantization

HydraLM is compatible with standard PyTorch post-training
quantization libraries. We have tested:

- **bitsandbytes 4-bit NF4**: ~3.4× memory reduction, <1% quality
  drop at 160M.
- **GPTQ 4-bit** via `auto_gptq`: similar memory / quality tradeoff,
  ~10% faster inference because it fuses the dequant kernel.
- **INT8 weight-only via `torchao`**: ~2× memory reduction, ~5%
  faster inference, negligible quality drop.

Quantization hooks live in the downstream libraries, not in this
repo — we keep HydraLM in pure PyTorch so every quantization stack
that speaks `nn.Linear` works out of the box.

## Serving

For high-throughput serving:

1. Use `compile_for_inference` with a fixed batch size per replica.
2. Pin the model to a single GPU and use CPU-side request batching
   with a queue — HydraLM's per-step compute is small enough that
   batching 4–16 requests saturates a single A100.
3. If using speculative decoding, keep the draft model co-resident
   on the same GPU; the inter-GPU synchronization cost dominates
   otherwise.

A minimal FastAPI example that wires all three together is in
`scripts/online_learning_demo.py`.

See `docs/api.md` for exact function signatures and
`docs/faq.md` for common deployment issues.
