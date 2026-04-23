# Deployment Guide

This document covers the deployment-facing APIs that exist in the current
HydraLM codebase: loading raw training checkpoints, generating with the
native recurrent path, using `StreamingEngine`, speculative decoding,
the compiled batched decoder, and the HuggingFace-shaped adapter.

## Loading a native HydraLM checkpoint

`HydraLM` itself does **not** currently expose `from_pretrained`. The
training stack writes `.pt` checkpoints containing model, optimizer,
scheduler, and step state, so the native reload flow is:

```python
import torch
from hydralm import HydraConfig, HydraLM

cfg = HydraConfig(
    vocab_size=32_000,
    d_model=768,
    n_layers=12,
    n_heads=12,
)

model = HydraLM(cfg)
ckpt = torch.load("checkpoints/step_10000.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
model = model.eval()
```

Important: the trainer checkpoint does not save `HydraConfig` for you, so
you must recreate the same config (or save it separately in your own
training pipeline).

## Single-shot generation

Use `hydralm.generate` for standard autoregressive generation:

```python
import torch
from hydralm import generate

input_ids = torch.tensor([[1, 2, 3, 4]])
out = generate(
    model,
    prompt_ids=input_ids,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
)
```

`generate` uses HydraLM's recurrent `step` path internally, so per-token
memory stays bounded by the model state rather than growing with output
length.

## Streaming generation

For long-context prefill and chunked processing, use `StreamingEngine`:

```python
from hydralm.streaming import StreamingEngine

engine = StreamingEngine(model, chunk_size=4096)

# Prefill-only pass that keeps memory bounded and returns stats.
stats = engine.process(long_input_ids)
print(stats.summary())

# Prefill the prompt and continue decoding.
generated = engine.extend_and_generate(
    prompt=long_input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_k=50,
)
```

If you need logits per chunk instead of a summary or decoded tokens, use
`engine.stream(token_chunks)`.

## Speculative decoding

HydraLM ships an exact speculative decoding implementation in
`hydralm.spec_decoding`.

### External draft model

```python
from hydralm import speculative_generate

generated, stats = speculative_generate(
    target=target_model,
    draft=draft_model,
    prompt_ids=input_ids,
    max_new_tokens=256,
    k=4,
)
```

### MTP self-drafting

If the checkpoint was trained with `cfg.mtp_depth > 0`, the model can
serve as both draft and target:

```python
generated, stats = speculative_generate(
    target=model,
    draft=model,
    prompt_ids=input_ids,
    max_new_tokens=256,
    k=model.cfg.mtp_depth,
)
```

The speculative decoder returns `(generated_ids, stats)`, where `stats`
tracks proposals, acceptances, and rounds.

## Retrieval Attention and bounded-memory serving

Retrieval layers keep a growing chunk bank during streaming inference. If
you need bounded-memory serving over very long streams, pair the attention
path with `CompressiveMemory`:

```python
from hydralm import CompressiveMemory

mem = CompressiveMemory(
    head_dim=model.cfg.head_dim,
    n_heads=model.cfg.n_heads,
    exact_window=model.cfg.retrieval_chunk_size,
    compress_every=4,
    n_summary=256,
)
```

`CompressiveMemory` is framework-agnostic: it transforms raw `(B, H, L, Dh)`
key/value streams, so you can integrate it anywhere you already manage
attention K/V tensors.

## Compiled batched decoding

For low-latency batched token generation, use `CompiledDecoder` and
`Request` from `hydralm.deploy`:

```python
from hydralm.deploy import CompiledDecoder, Request

decoder = CompiledDecoder(model, compile=True)
reqs = [
    Request(prompt=input_ids[0], max_new_tokens=64, temperature=0.8),
]
decoded = decoder.decode(reqs)
```

Key behavior:

- `prefill(reqs)` runs the prompt pass per request and stashes recurrent state
- `step_batch(reqs)` advances every active request by one token
- `decode(reqs)` runs prefill plus the full decode loop

Batching constraint: requests must have the same prompt length at prefill
time so the SWA caches line up cleanly across the batch dimension. In
practice that means bucketing requests by prompt length.

## HuggingFace-shaped adapter

If you want a `transformers`-style surface, use `HydraLMForCausalLM`:

```python
from hydralm.deploy import HydraLMForCausalLM

hf_model = HydraLMForCausalLM(cfg)
hf_model.model.load_state_dict(model.state_dict())
hf_model.save_pretrained("hf/hydralm-160m")

reloaded = HydraLMForCausalLM.from_pretrained("hf/hydralm-160m")
```

This adapter supports:

- `forward(...)` returning a `CausalLMOutput`-compatible object
- `generate(...)` delegating to HydraLM's native recurrent generator
- `save_pretrained(...)` / `from_pretrained(...)` for adapter-native checkpoints
- embedding resize/tie helpers expected by downstream tooling

`HydraLMForCausalLM.from_pretrained(...)` reloads checkpoints produced by
`HydraLMForCausalLM.save_pretrained(...)`; it is not a generic loader for
arbitrary trainer `.pt` checkpoints.

## Quantization

HydraLM stays in pure PyTorch, so standard post-training quantization stacks
that operate on `nn.Linear` modules can be layered on top. The repository
does not currently expose a first-party quantization wrapper.

## Practical serving notes

1. Use `generate` for the simplest path.
2. Use `StreamingEngine` when the prompt is large or comes in chunks.
3. Use `CompiledDecoder` when you want low-latency batched token serving.
4. Bucket requests by prompt length before `CompiledDecoder.prefill`.
5. Use the HF adapter only when you specifically need a Transformers-shaped
   wrapper or adapter-native save/load behavior.

See `docs/api.md` for exact signatures, `docs/retrieval.md` for the long-range
stack, and `docs/faq.md` for troubleshooting.
