# HydraLM API reference

This document is the canonical reference for every public symbol in
HydraLM. "Public" means anything re-exported from
`hydralm/__init__.py` or from a sub-package `__init__.py`
(`hydralm.modules`, `hydralm.kernels`, `hydralm.optim`,
`hydralm.memory`, `hydralm.training`, `hydralm.eval`, `hydralm.deploy`,
`hydralm.baselines`, `hydralm.streaming`, `hydralm.spec_decoding`,
`hydralm.utils`). Internal helpers with leading underscores are not
part of the contract and may change without notice.

Every example in this file runs on CPU with only `torch` installed
unless explicitly stated otherwise.

- [Package root (`hydralm`)](#package-root-hydralm)
  - [`HydraConfig`](#hydraconfig)
  - [`HydraLM`](#hydralm)
  - [`generate`](#generate)
  - [`speculative_generate` / `SpecDecodingStats`](#speculative_generate--specdecodingstats)
  - [`Muon` / `HybridMuonAdamW` / `build_hybrid_optimizer`](#muon--hybridmuonadamw--build_hybrid_optimizer)
- [`hydralm.streaming`](#hydralmstreaming)
- [`hydralm.modules`](#hydralmmodules)
- [`hydralm.kernels`](#hydralmkernels)
- [`hydralm.memory`](#hydralmmemory)
- [`hydralm.training`](#hydralmtraining)
- [`hydralm.eval`](#hydralmeval)
- [`hydralm.deploy`](#hydralmdeploy)
- [`hydralm.baselines`](#hydralmbaselines)
- [`hydralm.utils`](#hydralmutils)
- [State shapes](#state-shapes)

---

## Package root (`hydralm`)

The package root re-exports the symbols most users need:

```python
import hydralm
hydralm.__version__                   # "0.2.0"
hydralm.HydraConfig
hydralm.HydraLM
hydralm.generate
hydralm.speculative_generate
hydralm.SpecDecodingStats
hydralm.Muon
hydralm.HybridMuonAdamW
hydralm.build_hybrid_optimizer
```

### `HydraConfig`

```python
from hydralm import HydraConfig
```

Dataclass describing a HydraLM model. Every field has a sensible
default; you only need to override what you care about.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `vocab_size` | `int` | `32_000` | Size of the embedding matrix. |
| `pad_token_id` | `int` | `0` | Used for `nn.Embedding(padding_idx=...)`. |
| `d_model` | `int` | `768` | Residual stream width. |
| `n_layers` | `int` | `12` | Total number of `HydraBlock`s. |
| `n_heads` | `int` | `12` | Shared between DeltaNet and SWA blocks. |
| `head_dim` | `int \| None` | `None` | Auto-set to `d_model // n_heads` when `None`. |
| `mlp_mult` | `float` | `8/3` | SwiGLU hidden multiplier (LLaMA default). |
| `mlp_multiple_of` | `int` | `64` | Round the SwiGLU hidden size up to this multiple. |
| `dn_short_conv_kernel` | `int` | `4` | Short-conv kernel inside Gated DeltaNet. |
| `dn_use_gate` | `bool` | `True` | Enable SiLU output gate. |
| `dn_chunk_size` | `int` | `64` | Chunk length used by `delta_rule_chunkwise`. |
| `dn_norm_qk` | `bool` | `True` | L2-normalise queries / keys. Keep on for stability. |
| `swa_window` | `int` | `512` | Sliding-window attention span. |
| `swa_rope_base` | `float` | `10_000.0` | RoPE base frequency. |
| `layer_types` | `Sequence[Literal["deltanet","swa"]] \| None` | `None` | Manual block schedule. When `None`, one SWA is inserted every `swa_every` layers. |
| `swa_every` | `int` | `4` | Cadence of SWA layers in the default schedule. |
| `rms_eps` | `float` | `1e-5` | Epsilon in the `RMSNorm` denominator. |
| `tie_embeddings` | `bool` | `True` | Share weights between `embed` and the LM head. |
| `initializer_range` | `float` | `0.02` | Std of the normal initialiser. |
| `max_position_embeddings` | `int` | `1_048_576` | Soft upper bound, only enforced by tokenizer pipelines. |

**Computed attributes** (set in `__post_init__`):

- `layer_types`: tuple of length `n_layers`.
- `head_dim`: defaults to `d_model // n_heads` when omitted.

**Properties / methods:**

- `n_swa_layers: int` — count of `"swa"` entries in `layer_types`.
- `n_dn_layers: int` — count of `"deltanet"` entries.
- `summary() -> str` — single-line human description, used by scripts.
- `to_dict() -> dict` — JSON-serialisable dict suitable for
  `save_pretrained`. Tuples become lists so the resulting dict
  round-trips through `json.dumps`.

**Example — a 128 M parameter 1 M-token-context config:**

```python
cfg = HydraConfig(
    vocab_size=32_000,
    d_model=1024,
    n_layers=20,
    n_heads=16,
    swa_window=1024,
    swa_every=4,                # 5 SWA layers out of 20
    dn_chunk_size=128,
)
print(cfg.summary())
# HydraConfig(d_model=1024, n_layers=20, heads=16x64, DN=15, SWA=5 @ window=1024)
```

---

### `HydraLM`

```python
from hydralm import HydraLM
model = HydraLM(cfg)
```

The top-level language model. Roughly:

```
embed  ->  N x HydraBlock  ->  RMSNorm  ->  tied LM head
```

**Forward (parallel, O(N) over tokens):**

```python
out = model(input_ids, state=None, return_state=False)
# out: {"logits": FloatTensor(B, N, vocab), "state": list | None}
```

- `input_ids`: `LongTensor(B, N)`.
- `state`: optional list of per-layer caches carried over from a
  previous chunk. See [State shapes](#state-shapes). When `None`, the
  model initialises clean state (zeros for DeltaNet, empty ring buffers
  for SWA).
- `return_state`: when `True`, the returned dict's `"state"` field is
  the list of post-step per-layer caches. Required for chunked
  streaming prefill.

**Step (recurrent, O(1) per token):**

```python
logits_t, state = model.step(input_ids_t, state)
# input_ids_t : LongTensor(B,)
# logits_t    : FloatTensor(B, vocab)
# state       : list[dict]
```

Used by `generate`, `StreamingEngine.extend_and_generate`, and
`speculative_generate`. Under `@torch.no_grad()` — do not call in
training.

**Other methods:**

- `num_parameters(exclude_embedding: bool = False) -> int`.
- Standard `nn.Module` machinery (`state_dict`, `load_state_dict`,
  `train`, `eval`, `to(device, dtype)`, etc.).

**Example — train / eval / decode parity:**

```python
import torch
from hydralm import HydraConfig, HydraLM, generate

cfg = HydraConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = HydraLM(cfg).eval()

prompt = torch.randint(0, 256, (2, 16))
with torch.no_grad():
    out = model(prompt)
    logits = out["logits"]             # (2, 16, 256)
    completions = generate(model, prompt, max_new_tokens=8)   # (2, 24)
```

---

### `generate`

```python
from hydralm import generate

tokens = generate(
    model,
    prompt_ids,                 # LongTensor(B, N0)
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
)                               # LongTensor(B, N0 + T)
```

Batched autoregressive sampling. Internally:

1. **Prefill:** one parallel forward over the full prompt, returning
   the final recurrent state.
2. **Decode:** a loop that sampled one token per step via `model.step`,
   carrying the state forward.

Decoding memory is **independent of the generated length** — the only
state kept per layer is the fixed-size dict documented in
[State shapes](#state-shapes). `temperature <= 0` is treated as greedy
argmax. `top_k` and `top_p` can be combined; order of operations is
temperature → top-k → top-p → multinomial. `eos_token_id` short-circuits
finished rows with EOS padding.

---

### `speculative_generate` / `SpecDecodingStats`

```python
from hydralm import speculative_generate, SpecDecodingStats

tokens, stats = speculative_generate(
    target,                     # HydraLM — the large, authoritative model
    draft,                      # HydraLM — a cheaper model with matching vocab
    prompt_ids,                 # LongTensor(B, N0)
    max_new_tokens: int,
    k: int = 4,                 # speculation depth
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
)
```

Exact speculative decoding (Chen et al., 2023 / Leviathan et al.,
2023). Produces the exact target distribution token-for-token, while
requiring one target forward pass per `1 + accepted_len` tokens.

Unlike KV-cache Transformers, HydraLM can *snapshot and roll back* its
recurrent state in O(layers × state_size) independent of the sequence
length. This is done automatically on each round, so the algorithm is
correct under arbitrary rejection patterns.

`SpecDecodingStats` fields:

- `proposed: int` — total drafts attempted.
- `accepted: int` — drafts accepted.
- `rounds: int` — target forward passes.
- `acceptance_rate` (property) — `accepted / max(1, proposed)`.
- `mean_tokens_per_round` (property) — `(accepted + rounds) / rounds`,
  i.e. accepted-plus-bonus committed per target forward.

**Constraint:** `target.cfg.vocab_size == draft.cfg.vocab_size`.
The draft and target can otherwise differ in every dimension
(recommended: small `d_model`, shallow depth).

---

### `Muon` / `HybridMuonAdamW` / `build_hybrid_optimizer`

```python
from hydralm import Muon, HybridMuonAdamW, build_hybrid_optimizer
from hydralm.optim import zeropower_via_newton_schulz
```

Muon is a momentum optimiser whose update is pushed through a 5-step
Newton–Schulz iteration that maps it to the nearest semi-orthogonal
matrix, yielding ~1.3× sample efficiency over AdamW on LM pretraining
at the scales tested. See [`docs/training.md`](./training.md) for a
longer discussion.

**`Muon(params, lr=5e-3, momentum=0.95, nesterov=True, ns_steps=5,
weight_decay=0.0, adjust_lr=True)`** — for 2-D+ weight matrices only.
Passing a 1-D parameter raises `ValueError`. `adjust_lr` scales the
effective LR by `sqrt(max(1, fan_out / fan_in))` so that rectangular
matrices receive a comparable step size.

**`build_hybrid_optimizer(model, *, muon_lr=5e-3, muon_momentum=0.95,
muon_weight_decay=0.0, adamw_lr=3e-4, adamw_betas=(0.9, 0.95),
adamw_weight_decay=0.1, adamw_eps=1e-8, fused=None) -> HybridMuonAdamW`**
— the recommended entry point. Parameters are routed as:

- 2-D matrices in hidden layers → Muon.
- `embed.*`, `lm_head.*`, `wte.*` → AdamW (decay).
- `*.bias`, `RMSNorm` / 1-D gates → AdamW (no decay).

When `fused=None`, fused AdamW is enabled iff CUDA is present.

**`HybridMuonAdamW`** subclasses `torch.optim.Optimizer` and forwards
`step` / `zero_grad` to both internal optimisers. Its
`state_dict` / `load_state_dict` are nested dictionaries with `"muon"`
and `"adamw"` keys. Its `param_groups` is the concatenation of both
inner lists so `LambdaLR` applies to everything at once.

**`zeropower_via_newton_schulz(G, steps=5, eps=1e-7) -> Tensor`** — the
underlying orthogonalising primitive. Exposed for testing and for users
who want to build their own optimiser variants.

---

## `hydralm.streaming`

```python
from hydralm.streaming import StreamingEngine, StreamStats
```

`StreamingEngine(model, chunk_size=1024, device="cpu", dtype=torch.float32, keep_logits=False)`
feeds an arbitrarily long token stream through `model.forward` chunk
by chunk, carrying state between chunks. Peak memory is
O(state) — independent of the total length.

Three entry points:

- `process(tokens, progress_every=0) -> StreamStats` — run a `(B, N)`
  tensor through the model, discard the logits, return only stats.
- `stream(token_chunks) -> Iterator[(logits, stats)]` — yields
  per-chunk logits so the caller can process them incrementally.
- `extend_and_generate(prompt, max_new_tokens, temperature=1.0, top_k=None) -> Tensor`
  — chunked prefill over a possibly million-token prompt followed by
  standard autoregressive decoding.

`StreamStats` exposes `tokens_processed`, `chunks_processed`,
`peak_state_bytes`, `last_state_bytes`, `elapsed_seconds`, and a
`summary()` string.

---

## `hydralm.modules`

Low-level building blocks. Users who only want to run models can
ignore this module; authors who want to compose custom blocks should
prefer these over reimplementing them.

| Name | Role |
| --- | --- |
| `HydraBlock(cfg, layer_idx)` | Pre-norm residual unit dispatching to `GatedDeltaNet` or `SlidingWindowAttention`. |
| `GatedDeltaNet(d_model, n_heads, head_dim, short_conv_kernel=4, chunk_size=64, use_gate=True, norm_qk=True)` | Linear-time associative-memory mixer. |
| `SlidingWindowAttention(d_model, n_heads, head_dim, window=512, rope_base=10_000.0)` | Causal attention bounded to the last `window` tokens. |
| `SwiGLU(d_model, mult=8/3, multiple_of=64)` | Gated FFN. |
| `RMSNorm(dim, eps=1e-5)` | RMS layer norm (weight-only). |
| `RotaryEmbedding(head_dim, base=10_000.0)` | RoPE helper used by SWA only. |
| `ShortConv(dim, kernel_size=4)` | Depth-wise causal convolution with streaming cache. |

Every mixer and the short convolution follow the same two-method
contract:

```python
y,        state = mixer(x,   state=None)    # parallel / chunked
y_t,      state = mixer.step(x_t, state)    # single-token O(1)
```

The chunked and recurrent forms are numerically identical up to fp32
rounding (verified by `tests/test_streaming.py` and
`tests/test_equivalence.py`).

---

## `hydralm.kernels`

```python
from hydralm.kernels import (
    delta_rule_reference,
    delta_rule_recurrent,
    delta_rule_chunkwise,
)
```

Three implementations of the gated delta-rule recurrence. All return
`(out, final_state)` with shapes:

- `q`, `k`: `(B, H, N, Dk)`
- `v`: `(B, H, N, Dv)`
- `alpha`, `beta`: `(B, H, N)`
- `out`: `(B, H, N, Dv)`
- `final_state`: `(B, H, Dv, Dk)`

| Kernel | Use | Cost |
| --- | --- | --- |
| `delta_rule_reference` | Correctness reference. Explicit per-token scan. | `O(N·Dv·Dk)` |
| `delta_rule_recurrent` | Streaming inference. `torch.jit.script`ed tight loop. | `O(N·Dv·Dk)` |
| `delta_rule_chunkwise` | Training. `(C, C)` intra-chunk matmul + 1 inter-chunk update per chunk. | `O(N·(C·Dv + Dv·Dk))` |

All three take an optional `initial_state` (shape
`(B, H, Dv, Dk)`) so chunk-wise streaming is mathematically
equivalent to a single-shot forward over the concatenated input.

---

## `hydralm.memory`

```python
from hydralm.memory import FactBank, FactBankStats
```

`FactBank` is a zero-gradient associative memory backed by exactly the
same recurrence as `GatedDeltaNet`. Useful for RAG-style test-time
learning, programmatic fact overwrites, and debugging what the linear
layer can store.

**Constructor:**

```python
FactBank(
    head_dim: int,
    n_heads: int = 1,
    alpha: float = 1.0,                       # in (0, 1]
    beta: float = 1.0,                        # in (0, 1]
    head_init: Literal["rotation", "identity"] = "rotation",
    normalize_keys: bool = True,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
)
```

**Methods:**

- `memorize(keys: (N, D), values: (N, D)) -> FactBankStats` — write
  facts with the exact gated-delta recurrence. Destructive on
  duplicate keys.
- `recall(keys: (N, D), head_reduce: {"mean","first","stack"} = "mean") -> Tensor`
  — retrieve the associated values.
- `retrieval_accuracy(keys, values, metric={"cosine","mse"}) -> dict`
  — returns `cosine`, `cosine_min`, `mse`, `argmax_accuracy` against
  ground-truth `(keys, values)` pairs.
- `reset() -> None` — zero the state.
- `state_bytes: int` (property) — fixed-size footprint, independent of
  how many facts have been written.
- `num_facts_written: int` (property).

All writes and reads execute under `@torch.no_grad()`; the
`FactBankStats.writes_require_grad` flag is there so test suites can
assert this invariant.

---

## `hydralm.training`

```python
from hydralm.training import Trainer, TrainingConfig
```

**`TrainingConfig`** — dataclass with the fields used by `Trainer`:

| Field | Default | Notes |
| --- | --- | --- |
| `steps` | `10_000` | Total optimiser steps. |
| `batch_size` | `8` | Per-rank. |
| `grad_accum` | `1` | Micro-batches per optimiser step. |
| `lr` | `3e-4` | AdamW base LR; also the anchor for the cosine schedule. |
| `min_lr` | `3e-5` | Floor of the cosine decay. |
| `warmup_steps` | `500` | Linear warmup. |
| `weight_decay` | `0.1` | AdamW decay on 2-D params. |
| `betas` | `(0.9, 0.95)` | AdamW betas. |
| `grad_clip` | `1.0` | Max grad-norm. |
| `log_every` / `eval_every` / `save_every` | `50` / `1000` / `2000` | Step cadences. |
| `checkpoint_dir` | `"./checkpoints"` | Saved as `<dir>/step_<N>.pt`. |
| `mixed_precision` | `"auto"` | `"auto"` picks bf16 on Ampere+, fp16 otherwise; `"bf16"`, `"fp16"`, `"none"` are also valid. |
| `use_fsdp` | `True` | Ignored if `WORLD_SIZE <= 1`. |
| `grad_checkpoint` | `False` | Wraps every `HydraBlock.forward` in `torch.utils.checkpoint`. |
| `compile` | `False` | `torch.compile(model)` before training. |
| `seed` | `0` | |
| `log_fn` | `None` | Callable receiving a dict per `log_every`. |
| `optimizer` | `"adamw"` | or `"muon"` to use `build_hybrid_optimizer`. |
| `muon_lr`, `muon_momentum` | `5e-3`, `0.95` | Used only when `optimizer="muon"`. |

**`Trainer(model, cfg)`** — holds the model, optimiser, scheduler, AMP
context, and grad scaler. Exposes:

- `fit(data, on_eval=None)` — main loop.
  - `data`: iterable / iterator of `(input_ids, labels)` tensors. The
    trainer re-starts the iterator on exhaustion.
  - `on_eval(model, step)`: optional rank-0 callback every
    `cfg.eval_every` steps.
- `step: int` — number of optimiser steps taken so far.

The loss is causal-LM cross-entropy with label shift-by-one and
`ignore_index=-100`. Under `WORLD_SIZE > 1` the trainer initialises a
NCCL process group, wraps `model` in FSDP with a
`HydraBlock`-granularity auto-wrap policy, and uses `DistributedSampler`-style
sharding on the user side.

---

## `hydralm.eval`

A small collection of benchmarks that the claim harness is built on.

- `mqar`:
  - `MQARConfig`, `make_mqar_batch`, `evaluate_mqar`, `train_mqar` —
    generators, trainer, and scorer for the multi-query associative
    recall task of Arora et al. (2024).
- `long_context`:
  - `LongContextConfig`, `make_needle_batch`, `evaluate_needle` —
    needle-in-a-haystack generator and scorer.
- `online_learning`:
  - `evaluate_memorization`, `evaluate_capacity_curve`,
    `evaluate_interference`, `evaluate_overwrite`,
    `kv_cache_memory_comparison` — `FactBank`-centric evaluations.
- `claims`:
  - `ClaimResult`, `ClaimReport`, `paired_claim_config`,
    `check_claim_1_linear_complexity`, `check_claim_2_lossless_mqar`,
    `check_claim_3_constant_state`, `check_claim_4_cost_reduction`,
    `check_claim_5_drop_in`, `check_claim_6_online_learning`,
    `ALL_CLAIMS`, `run_all_claims`.

See [`docs/evaluation.md`](./evaluation.md) and
[`docs/claims.md`](./claims.md) for usage.

---

## `hydralm.deploy`

```python
from hydralm.deploy import (
    HydraLMForCausalLM,
    CausalLMOutput,
    CompiledDecoder,
    Request,
)
```

**`HydraLMForCausalLM(config)`** — a `transformers`-shaped adapter that
exposes `forward(input_ids, labels=None, attention_mask=None,
past_key_values=None, use_cache=False, return_dict=True, **kwargs) ->
CausalLMOutput`. `past_key_values` carries HydraLM's recurrent state
(not a KV cache, despite the name — the field exists for API parity
with `transformers`). Supports `get_input_embeddings`,
`set_input_embeddings`, `get_output_embeddings`, `tie_weights`,
`resize_token_embeddings`, `save_pretrained`, `from_pretrained`, and a
`.generate(...)` method that delegates to `hydralm.generate`.

`save_pretrained(dir)` writes `pytorch_model.bin` plus
`hydra_config.json`; `from_pretrained(dir, **overrides)` reloads them
and restores the exact `layer_types` schedule.

**`CompiledDecoder(model, compile=True)`** — low-latency batched
decoder that wraps `model.step` in
`torch.compile(mode="reduce-overhead")` and stacks per-request state
across the batch dimension.

- `prefill(reqs: list[Request])` — run the parallel prefill for each
  request and stash its final state.
- `step_batch(reqs)` — advance one token for every not-done request.
- `decode(reqs)` — prefill + step-loop-until-done; returns a list of
  `LongTensor`s (prompt + generated tokens per request).

**`Request`** — in-flight decoding job: `prompt`, `max_new_tokens`,
`temperature`, `top_k`, `top_p`, `eos_token_id`. The decoder
populates `produced`, `state`, and `done` during processing.

**Batching constraint:** at prefill time requests must be bucketed by
prompt length so the SWA `k_cache`/`v_cache` tensors align along the
batch axis. The DeltaNet state is shape-invariant across requests and
imposes no constraint. A paged-state follow-up that lifts this
restriction is tracked in the [roadmap](./roadmap.md).

---

## `hydralm.baselines`

```python
from hydralm.baselines import DenseTransformer, flops
```

`DenseTransformer(cfg)` — a reference dense Transformer that mirrors
HydraLM's public interface (`model(input_ids) -> {"logits": ...}`,
`model.num_parameters()`, `model.cfg`). Used by the benchmarks and the
claim gates for like-for-like comparisons.

`flops` — a module of closed-form FLOP / memory / dollar functions
shared between the claim gates and `scripts/cost_analysis.py`. The
single source of truth for "what should the theoretical cost be at
length `N`?".

---

## `hydralm.utils`

```python
from hydralm.utils import seed_everything, count_parameters, human_bytes
```

- `seed_everything(seed: int = 0)` — seeds Python `random`, NumPy, and
  PyTorch (including CUDA if available).
- `count_parameters(model, trainable_only: bool = True) -> int`.
- `human_bytes(n: float) -> str` — e.g. `human_bytes(1.5e9) == "1.40 GiB"`.

---

## State shapes

`HydraLM.forward` with `return_state=True` and `HydraLM.step` both
return a list whose length equals `cfg.n_layers`. Each entry is a dict
whose shape depends on the block type:

**DeltaNet layer** (`cfg.layer_types[i] == "deltanet"`):

| Key | Shape | Notes |
| --- | --- | --- |
| `conv_cache` | `(B, 3 * n_heads * head_dim, dn_short_conv_kernel - 1)` | Rolling short-conv state. `None` before first call if `kernel == 1`. |
| `S` | `(B, n_heads, head_dim, head_dim)` | The recurrent delta-rule state. |

**SWA layer** (`cfg.layer_types[i] == "swa"`):

| Key | Shape | Notes |
| --- | --- | --- |
| `k_cache` | `(B, n_heads, min(pos, swa_window), head_dim)` | Already-rotated keys. |
| `v_cache` | `(B, n_heads, min(pos, swa_window), head_dim)` | |
| `pos` | `int` | Absolute position of the next token (scalar). |

Every tensor in a state dict is independent of the total sequence
length processed so far, which is why HydraLM's inference memory is
O(state) rather than O(N).
