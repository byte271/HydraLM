# Glossary

A short reference for the terminology and acronyms used throughout the
HydraLM codebase and documentation. Entries are alphabetized.

### AdamW

Adam with decoupled weight decay (Loshchilov & Hutter, 2017). The
default optimizer for 1-D and 0-D parameters in
`HybridMuonAdamW`.

### bfloat16 (bf16)

A 16-bit floating-point format with the same 8-bit exponent as fp32
but only 7 bits of mantissa. Trades precision for range, making it
the preferred mixed-precision dtype on Ampere+ GPUs.

### CUDA graph

A recorded sequence of CUDA operations that can be replayed with near
zero launch overhead. Enabled implicitly by
`compile_for_inference(..., mode="reduce-overhead")`.

### Delta rule

An online associative-memory update `S ← S + k (v − Sᵀk)ᵀ`
(Widrow & Hoff, 1960) that overwrites rather than accumulates. The
gated variant scales the update by learned `α, β ∈ (0,1)` per token.

### DeltaNet / GDN

A sequence-mixing layer that applies the gated delta rule along the
time axis. Constant-memory, O(N) FLOPs. In HydraLM,
`hydralm.modules.gated_deltanet.GatedDeltaNet`.

### Fact bank

An external, explicit, non-differentiable key-value store queried at
generation time. See `docs/fact-bank.md`.

### FLOPs per token

Floating-point operations required to process a single token through
one forward pass. For Transformers this scales with N; for DeltaNet
it is constant.

### head_dim

Per-head hidden size. In HydraLM, `cfg.head_dim = cfg.d_model /
cfg.n_heads` unless explicitly set.

### HybridMuonAdamW

HydraLM's default optimizer: Muon for 2-D linear weights, AdamW for
everything else. See `docs/training.md`.

### KV cache

The cached key/value projections at every Transformer layer, keyed by
position. Grows linearly with sequence length in a standard
Transformer; replaced by an O(1) state matrix in DeltaNet and a
rolling buffer of size `swa_window` in SWA.

### Linear attention

Attention whose per-step cost is constant in the sequence length,
obtained by linearizing the softmax kernel to an inner product
(Katharopoulos et al., 2020).

### MFU

Model FLOPs Utilization. Fraction of a GPU's theoretical peak FLOPs
that the training loop actually uses. Typical HydraLM training on
A100 hits ~85% MFU with `compile=True`.

### MQAR

Multi-Query Associative Recall. A synthetic benchmark in which a
sequence of `(key, value)` pairs is followed by queries; the model
must retrieve the correct value for each. See `docs/evaluation.md`.

### Muon

An optimizer that uses Newton-Schulz orthogonalization on the
momentum-smoothed gradient before the weight update (Jordan et al.,
2024). Effective for 2-D matrix parameters.

### Needle-in-a-Haystack

A long-context evaluation that inserts a single fact ("the needle")
at varying depths inside a long distractor corpus and asks the model
to retrieve it. See `docs/evaluation.md`.

### Pre-norm

A residual-block layout where normalization is applied to the input
of each sublayer rather than the output. Produces more stable
training dynamics than the original post-norm Transformer.

### RMSNorm

Root Mean Square Layer Normalization (Zhang & Sennrich, 2019). The
only normalization used in HydraLM.

### RoPE

Rotary Position Embedding (Su et al., 2021). Encodes positions as
rotations of query/key vector pairs; the only positional encoding
used in HydraLM's SWA layers.

### Short convolution

A small causal 1-D depthwise convolution (kernel size 3–5) applied to
the `(q, k, v)` projections before the DeltaNet recurrence. Captures
short-range patterns that would otherwise contaminate the recurrent
state.

### Speculative decoding

An inference-time technique that uses a small draft model to propose
k tokens, which the larger target model verifies in a single parallel
pass (Leviathan et al., 2023; Chen et al., 2023).

### State dict (HydraLM)

At inference time, a `List[Dict[str, Tensor]]` — one entry per layer
— holding the layer-specific recurrent state. Distinct from
PyTorch's module `state_dict()`, which holds parameters.

### SwiGLU

The Swish-Gated Linear Unit MLP variant (Shazeer, 2020). Three
linear projections: `up`, `gate`, `down`; output is `down(silu(gate)
* up)`.

### SWA

Sliding-Window Attention. Softmax attention restricted to a causal
window of size `W`. See `hydralm.modules.sliding_window`.

### Target / draft model (speculative decoding)

The *target* is the large, high-quality model whose outputs you want.
The *draft* is a small model whose proposals the target verifies.

### TinyShakespeare

A 1.1 MB text file of Shakespeare's collected works, bundled as
`data/tinyshakespeare.txt`. Used as the smoke-test corpus for
`scripts/train_tiny.py`.

### Token-per-second (tok/s)

The standard throughput metric for autoregressive generation. In this
repository always measured with `torch.cuda.Event`, batch size 1,
after a 20-token warmup.

### W (`swa_window`)

The window size of the sliding-window attention layers, in tokens.
Default is 256.
