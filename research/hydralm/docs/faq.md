# Frequently Asked Questions

## General

### Is HydraLM production-ready?

HydraLM is a **research reference implementation**. The code is
clean, tested, and numerically stable, but it has not been hardened
for multi-tenant serving and has no first-party quantization path.
For production, we recommend either:

- Training a model with HydraLM and exporting it to Hugging Face via
  `hydralm.deploy.hf_adapter.export_to_hf`, then serving it with
  `vllm` / `sglang` / `TGI`.
- Using the exposed nn.Modules directly inside your own framework
  (Lightning, nanotron, torchtitan) and treating HydraLM as a layer
  library rather than a training stack.

### Why hybrid? Why not just Transformer or just Mamba?

See `docs/architecture.md` §1 and `docs/theory.md` §5. The short
answer: pure linear-attention models have capacity-bounded state
that caps their recall; pure softmax Transformers have
unbounded-memory inference cost. A small number of SWA layers
restores local recall at negligible compute cost.

### Does HydraLM use flash-attention?

Not yet. The SWA layer is a plain PyTorch implementation that is
*correct* but not *maximally fast*. Flash-attention 3 support is on
the roadmap (`docs/roadmap.md`). For current inference throughput
this is rarely a bottleneck because the window is small (256 tokens
by default), so the attention is already cache-friendly.

---

## Training

### I get NaNs after a few hundred steps. What's wrong?

In order of likelihood:

1. **fp16 in the delta-rule path.** The accumulator `S` must be fp32.
   Do not wrap the model in a `torch.autocast(dtype=torch.float16)`
   context. Use `precision="bfloat16"` or `"float32"` in
   `TrainerConfig`.
2. **Learning rate too high at warmup end.** Default is 3e-4; try
   1e-4 if you have a small batch size.
3. **Gradient clipping disabled.** Always use `grad_clip >= 1.0`.

### Training is much slower than I expected.

Checklist:

- Did you set `compile=True` in `TrainerConfig`? First iteration will
  be slow; steady state should be ~85% MFU on A100.
- Are you on PyTorch ≥ 2.4? Earlier versions miss essential
  `torch.compile` fixes for the delta-rule path.
- Are you using `precision="bfloat16"`? fp32 is ~2.5× slower.
- Are you packing sequences to the configured `seq_len`? Short
  sequences leave the GPU idle.

### Can I fine-tune a published HydraLM checkpoint?

Yes — `HydraLM.from_pretrained(...)` returns an ordinary `nn.Module`
that you can plug into any training loop. The architecture is
unchanged during fine-tuning, so no special handling is needed.

### What is the recommended batch size?

At `d_model=768`, `seq_len=2048`, bfloat16, on an A100 80GB: global
batch of 512k tokens (e.g. 64 seqs × 2048 × 4 grad accum × 1 GPU) is
a good default. Scale linearly with GPU count.

---

## Inference

### Why is the first token slow?

`torch.compile` plus CUDA-graph capture happens on the first call
inside `compile_for_inference`. Warm up by emitting a dummy token
before measuring.

### How do I clear state between requests in a server?

Pass `state=None` to `generate` / `stream` / `step_fn` for each new
request. State is not mutated by the functions you call; you get a
new state dict back that you can discard.

### Can I run a 160M HydraLM on CPU?

Yes, but you will not be happy. Expect ~1–2 tok/s because neither
the delta-rule kernel nor PyTorch's attention primitives are CPU-
optimized. A single consumer GPU (RTX 3060 or better) will be
≥50× faster.

### Does HydraLM support batched inference with different prompt
lengths?

Yes, via left-padding and an attention mask. For best throughput we
recommend bucketing requests into lanes of similar length and
compiling one `step_fn` per bucket.

---

## Deployment

### How do I quantize a HydraLM model?

Standard PyTorch quantization libraries work out of the box because
HydraLM is pure `nn.Module` / `nn.Linear`. See `docs/deployment.md` §
"Quantization" for tested combinations (bitsandbytes, GPTQ, torchao).

### Is the Hugging Face export two-way?

One-way only. `hydralm.deploy.hf_adapter.export_to_hf` writes a HF-
compatible checkpoint. There is no `from_hf` because the HF archive
contains metadata (tokenizer, generation config) we don't round-trip.
Use `HydraLM.from_pretrained` to reload a HydraLM-native checkpoint.

### What's the smallest useful model size?

`d_model=256`, `n_layers=6` (~15M params) is the smoke-test
configuration in `scripts/train_tiny.py`. It learns the
TinyShakespeare corpus in minutes but has no zero-shot ability. For
actual downstream use, 125M–400M is the useful range for research;
see `docs/benchmarks.md`.

---

## Fact bank

### How big can the fact bank be?

In GPU memory: bounded by `capacity * dim * dtype_bytes`. At
`capacity=1_000_000`, `dim=768`, fp16: ~1.5 GB. Beyond that, use
`bank.to_cpu()` and accept a latency hit, or swap in a FAISS index
via `bank.set_backend(...)`.

### Can I populate the bank in parallel across GPUs?

Yes, using any DDP pattern. The bank itself is not distributed
(single-GPU, single-process), but its `write()` method is thread-
safe — in a DDP setup, each rank can own a shard of the bank and
broadcast queries.

### Does the bank survive `torch.save` / `torch.load`?

Use `bank.state_dict()` and `bank.load_state_dict()` explicitly. A
raw `torch.save(bank)` saves device-specific tensors that may not
load cleanly on a different GPU topology.

---

## Contributing and help

### Where do I file a bug?

`https://github.com/byte271/hydralm/issues`. Please include your
PyTorch version, CUDA version, a minimal reproduction, and the
commit hash of HydraLM you are using (`hydralm.__version__`).

### How do I propose a new claim or benchmark?

Open a PR at `https://github.com/byte271/hydralm/pulls` that adds:

1. A script under `scripts/` that can be run by a stranger in under
   an hour.
2. A test under `tests/` that gates the pass threshold.
3. An entry in `docs/claims.md` that links the two.

See `CONTRIBUTING.md` for the full workflow.
