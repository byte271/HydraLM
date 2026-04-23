# Frequently Asked Questions

## General

### Is HydraLM production-ready?

HydraLM is a **research reference implementation**. The code is clean,
tested, and useful for experimentation, but it is not a full serving
platform. Think of it as:

- a model implementation
- a small training stack
- evaluation and reproducibility harnesses
- deployment-facing helpers such as `StreamingEngine`, `CompiledDecoder`,
  and `HydraLMForCausalLM`

### Why hybrid instead of pure Transformer or pure linear attention?

Because the tradeoff is the point:

- DeltaNet gives the backbone a fast recurrent path.
- SWA restores exact local recall.
- Retrieval Attention adds sparse long-range access when local recall is not enough.

The goal is to preserve the scaling benefits of linear-style sequence
modeling without giving up precise recall everywhere.

### How do I extract a specific fact from a very long context?

Enable Retrieval Attention:

```python
cfg = HydraConfig(
    ...,
    retrieval_every=3,
    retrieval_chunk_size=128,
    retrieval_top_k=8,
)
```

For very long streaming workloads, pair the retrieval path with
`CompressiveMemory` if you need bounded-memory serving behavior.

### When should I use DeltaNet, SWA, Retrieval, or the FactBank?

| Mechanism | Best for | Cost profile |
| --- | --- | --- |
| DeltaNet | Fast backbone dynamics and long-range gist | Recurrent, bounded state |
| SWA | Exact local recall within a fixed recent window | Window-bounded softmax |
| Retrieval | Pulling relevant distant chunks into view | Sparse long-range softmax |
| FactBank | Explicit out-of-band memory you write/query yourself | External key-value memory |

### What's the overhead of enabling MTP?

At training time, enabling `mtp_depth > 0` adds an auxiliary next-k
prediction head and produces `out["mtp_aux_loss"]` when
`compute_mtp=True`.

At inference time, there is no extra cost unless you explicitly use the
MTP path, for example during self-drafting speculative decoding.

## Training

### What trainer config class should I use?

Use `TrainingConfig` from `hydralm.training.trainer`, not `TrainerConfig`.

### Is Muon the default optimizer?

Not in the built-in trainer. `TrainingConfig.optimizer` defaults to
`"adamw"`. Set it to `"muon"` to use the hybrid Muon + AdamW path.

### How do I reload a native checkpoint?

Native trainer checkpoints are `.pt` files. Recreate the config, build
`HydraLM(cfg)`, then load `state["model"]`:

```python
state = torch.load("checkpoints/step_2000.pt", map_location="cpu", weights_only=True)
model = HydraLM(cfg)
model.load_state_dict(state["model"])
```

There is no `HydraLM.from_pretrained(...)` method in the current codebase.

### Does the trainer support distributed execution?

Yes. The built-in trainer can wrap the model in FSDP when launched under
distributed execution and `use_fsdp=True`.

## Inference and deployment

### What's the simplest inference path?

Use `hydralm.generate(...)` with a prompt tensor and `max_new_tokens`.

### How do I process a very long prompt?

Use `StreamingEngine`:

- `process(...)` for prefill-only stats
- `stream(...)` if you need per-chunk logits
- `extend_and_generate(...)` for prefill plus decode

### Why is the first compiled decode slower?

If you use `CompiledDecoder(model, compile=True)`, the first decode round
may pay compilation overhead. That is expected.

### How do I clear state between requests?

- `generate(...)` handles its own state internally.
- `StreamingEngine` starts fresh when you create a new engine or call it
  with a new prompt.
- `CompiledDecoder` stores state inside each `Request`, so starting a new
  request means creating a new `Request` object.

### Can I batch requests with different prompt lengths?

Not in a single `CompiledDecoder.prefill(...)` batch. Bucket requests by
prompt length before prefill so the SWA caches align cleanly.

### Is there a HuggingFace-compatible wrapper?

Yes: `HydraLMForCausalLM` in `hydralm.deploy`.

It supports:

- a `transformers`-shaped `forward(...)`
- recurrent `generate(...)`
- `save_pretrained(...)`
- `from_pretrained(...)`

### Is the HuggingFace adapter export two-way?

`HydraLMForCausalLM.from_pretrained(...)` reloads checkpoints written by
`HydraLMForCausalLM.save_pretrained(...)`. It is not a generic loader for
trainer `.pt` checkpoints.

## Fact bank

### What is the FactBank for?

`FactBank` is an explicit key-value memory you control programmatically.
Use it when you know what facts should be written and queried at inference
time, rather than relying only on learned in-context retrieval.

## Contributing

### Where do I file a bug?

[GitHub Issues](https://github.com/byte271/HydraLM/issues)

### How do I propose a new benchmark or claim?

Open a PR that adds:

1. a reproducible script under `scripts/`
2. the matching test coverage under `tests/`
3. the corresponding documentation in `docs/claims.md`

Pull requests go to:
[byte271/HydraLM](https://github.com/byte271/HydraLM/pulls)
