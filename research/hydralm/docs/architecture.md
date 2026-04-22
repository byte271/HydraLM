# HydraLM Architecture

HydraLM is a *hybrid* causal language model that interleaves two
complementary sequence mixers inside a standard pre-norm Transformer
backbone:

1. **Gated DeltaNet** — a linear-attention layer with a learned delta
   rule that gives constant-memory, O(N) recurrent inference.
2. **Sliding-Window Attention (SWA)** — a classical softmax attention
   layer restricted to the last `W` tokens, giving exact local recall
   with constant per-token cost.

Two layers types alternate according to the `cfg.layer_types`
schedule. The default schedule (`None`) alternates `deltanet` and
`swa` starting from the bottom, producing an N:1 ratio controllable by
replacing any entry in the list.

```
┌─────────────────────────────────────────────────────────┐
│                    Token embeddings                     │
└────────────────────────────┬────────────────────────────┘
                             │
   ┌─────────────────────────▼─────────────────────────┐
   │  Block 0  (deltanet)   RMSNorm → GDN → RMSNorm    │
   │                                     → SwiGLU MLP  │
   └─────────────────────────┬─────────────────────────┘
                             │
   ┌─────────────────────────▼─────────────────────────┐
   │  Block 1  (swa)        RMSNorm → SWA → RMSNorm    │
   │                                     → SwiGLU MLP  │
   └─────────────────────────┬─────────────────────────┘
                            ...
   ┌─────────────────────────▼─────────────────────────┐
   │  Block N-1             Final RMSNorm              │
   └─────────────────────────┬─────────────────────────┘
                             │
                       tied LM head
```

## Why hybrid?

Pure linear-attention models like RWKV, Mamba, or plain DeltaNet are
asymptotically attractive (O(N) training, O(1) inference state) but
lose *exact* recall of specific tokens — their state is a fixed-size
associative memory, not a KV cache. Pure softmax Transformers retain
exact recall but pay O(N²) training and O(N) inference-time memory.

Empirically, sprinkling a small number of SWA layers into a DeltaNet
stack recovers the recall gap almost entirely while preserving most of
the linear-attention cost profile. This aligns with the published
findings for **Samba**, **Gated DeltaNet**, **Jamba**, and **Zamba2**.

HydraLM's contribution is not any one of these ideas in isolation; it
is a clean, documented, test-covered reference implementation that
makes the hybrid tradeoff easy to study at sub-billion-parameter
scale.

## Block layout

Every block is a standard pre-norm residual:

```
x = x + mixer(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

The only difference between layer types is the `mixer` submodule:

| Layer type  | Mixer                          | State during inference |
| ----------- | ------------------------------ | ---------------------- |
| `deltanet`  | `GatedDeltaNet`                | Short-conv cache + `S` matrix |
| `swa`       | `SlidingWindowAttention`       | Rolling K/V cache of length W |

The MLP is a **SwiGLU** (Shazeer 2020) with `mlp_ratio * d` hidden
units rounded up to a multiple of `mlp_multiple_of`. The LM head and
the token embedding are **tied** — they share the same weight matrix.

## Gated DeltaNet in one paragraph

Each GDN layer projects the input `x` into three linear-attention
streams `(q, k, v)` and two per-token gates `(β, α)`:

- `β ∈ (0, 1)` is the *write strength* of the current token;
- `α ∈ (0, 1)` is the *decay* of the accumulated state.

A shared depthwise causal **1-D convolution** (`dn_short_conv_kernel`
taps) is applied to `q`, `k`, `v` before the recurrence. This short
conv is what lets the layer learn short-range patterns that the
recurrent state alone would spread too thin.

The recurrence itself is the gated delta rule:

```
S ← α · S + β · (v − Sᵀk) kᵀ
o  = S q
```

where `S` is the per-head state matrix of shape `(head_dim, head_dim)`.
The implementation in `hydralm.kernels.delta_rule` unrolls this along
the time dimension during training and supports step-wise updates
during inference.

## Sliding-window attention in one paragraph

SWA is ordinary multi-head attention with a causal mask whose active
window is `cfg.swa_window` tokens wide. Positions are mapped to
queries and keys through **RoPE**. The KV cache is stored as a
*rolling* buffer during inference so that memory usage stays
O(n_heads · W · head_dim) regardless of how many tokens have been
emitted. The implementation lives in `hydralm.modules.sliding_window`.

## Normalization, initialization, and dtypes

- **Normalization**: RMSNorm everywhere — before each mixer, before
  each MLP, and one final RMSNorm before the LM head.
- **Initialization**: linear layers use truncated normal with
  `std = initializer_range`; biases are zero; RMSNorm weights start at
  one. The short-conv is initialized to an identity delta so that the
  network starts as a clean linear-attention stack before training
  learns its filters.
- **Dtypes**: weights default to fp32 for numerical safety, but the
  model is trained and deployed under `torch.autocast('cuda',
  dtype=bfloat16)`. The delta-rule kernel runs the accumulator `S` in
  fp32 regardless of the activation dtype — see `docs/theory.md` for
  why this matters.

## Where to look in the source tree

| Concern                          | Entry point |
| -------------------------------- | ----------- |
| Model end-to-end                 | `hydralm/model.py` |
| Block scheduling                 | `hydralm/config.py` → `HydraConfig.layer_types` |
| Gated DeltaNet layer             | `hydralm/modules/gated_deltanet.py` |
| Delta-rule kernel                | `hydralm/kernels/delta_rule.py` |
| Sliding-window attention         | `hydralm/modules/sliding_window.py` |
| SwiGLU MLP                       | `hydralm/modules/swiglu.py` |
| RMSNorm                          | `hydralm/modules/rmsnorm.py` |
| RoPE                             | `hydralm/modules/rotary.py` |
| Short convolution                | `hydralm/modules/short_conv.py` |
| Generation loop & streaming      | `hydralm/generation.py`, `hydralm/streaming.py` |
| Speculative decoding             | `hydralm/spec_decoding.py` |
| Fact bank (external memory)      | `hydralm/memory/fact_bank.py` |

See `docs/theory.md` for a derivation of the delta rule and a
discussion of the recall/efficiency tradeoff.
