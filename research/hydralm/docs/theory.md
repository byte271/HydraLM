# Theory

This document gives a self-contained derivation of the **gated delta
rule** used in every HydraLM DeltaNet layer, the intuition for why
mixing it with sliding-window attention works, and the complexity
analysis that drives the benchmarks in `RESULTS.md`.

## 1. From softmax attention to linear attention

Recall softmax attention on a single head:

$$o_t = \sum_{s \le t} \frac{\exp(q_t^\top k_s)}{\sum_{u \le t} \exp(q_t^\top k_u)} \, v_s$$

The dependence on $t$ of *both* the numerator and the denominator is
what makes a KV cache grow linearly — every previous `k_s` is needed
again the next step.

Linearizing the softmax kernel to an inner product $\phi(q)^\top
\phi(k)$ (Katharopoulos et al., 2020) yields

$$o_t = \phi(q_t)^\top \underbrace{\sum_{s \le t} \phi(k_s) v_s^\top}_{S_t}$$

with an O(d × d) matrix state $S_t$ that is updated **additively**:

$$S_t = S_{t-1} + \phi(k_t) v_t^\top$$

This is O(1) per step, constant memory, but strictly *add-only*. Once
a (key, value) pair has been written, it cannot be overwritten, which
limits how well the layer can recall recently observed tokens versus
older ones.

## 2. The delta rule

Widrow & Hoff's classical delta rule solves the overwrite problem by
subtracting out the *predicted* value before adding the new one:

$$S_t = S_{t-1} + \phi(k_t) \bigl(v_t - S_{t-1}^\top \phi(k_t)\bigr)^\top$$

The term in parentheses is the "surprise" of the current key — the
difference between what the state would predict and what actually
arrived. The recurrence is still linear in time, and still
constant-memory, but it now implements an *online least-squares
associative memory*.

## 3. Gating

HydraLM's delta rule is **gated**: per-token scalars $\alpha_t,
\beta_t \in (0, 1)$ control write strength and decay:

$$S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot \phi(k_t) \bigl(v_t - S_{t-1}^\top \phi(k_t)\bigr)^\top$$

- $\beta_t \to 0$: the current token is ignored.
- $\beta_t \to 1$: the current token overwrites fully.
- $\alpha_t \to 1$: the state is persistent.
- $\alpha_t \to 0$: the state is reset — the layer becomes ephemeral.

Both gates are produced from $x_t$ by linear projections followed by a
sigmoid. The final output is

$$o_t = S_t \, \phi(q_t)$$

and in HydraLM we set $\phi(x) = x$ (no feature map), relying on the
short-conv and RoPE for positional encoding.

## 4. Numerical considerations

The accumulator $S_t$ is rank-bounded by `head_dim`, so it can in
principle store up to `head_dim` distinct associations cleanly. In
practice, when activations are in bfloat16:

- $S$ drifts quickly because bfloat16 has only 7 bits of mantissa.
- Small decays ($\alpha \to 0.99\ldots$) compound catastrophically.

HydraLM keeps $S$ in **fp32** inside the kernel regardless of the
surrounding autocast dtype. The inputs $q, k, v$ are cast to fp32
for the update and the output is cast back before the residual add.
This matches the recipe from the Gated DeltaNet paper and is critical
for stability at long context lengths.

## 5. Why mix DeltaNet with sliding-window attention?

Three well-known limitations of linear/recurrent layers:

1. **State capacity is bounded by `head_dim`.** A DeltaNet layer with
   `head_dim = 64` cannot hold more than 64 linearly-independent
   associations at once. Long sequences that probe many distinct
   facts (multi-query associative recall, needle-in-a-haystack) will
   degrade.
2. **Exact token recall is expensive.** Recovering the *exact* value
   of a specific past token requires the state to have memorized it
   losslessly — again, capacity-bounded.
3. **In-context learning asks for both.** Real workloads mix "fresh
   local context" (last paragraph) with "facts we introduced far
   back" (the definitions at the top of a document).

Sliding-window attention addresses (1) and (2) for the **local**
window: any query attends exactly to the last `W` tokens with
full-precision softmax. The DeltaNet state handles everything beyond
`W` with a lossy but constant-memory summary.

Empirically (see `docs/benchmarks.md`), a 1-SWA-per-4-DeltaNet
schedule closes most of the recall gap to a pure Transformer while
retaining ~80% of the linear-attention speedup at long context.

## 6. Complexity summary

Let $N$ = sequence length, $d$ = model dim, $W$ = SWA window,
$L_g, L_s$ = number of DeltaNet / SWA layers, $h$ = head count.

| Quantity                | Transformer | DeltaNet    | HydraLM                               |
| ----------------------- | ----------- | ----------- | ------------------------------------- |
| Training FLOPs / token  | $O(N d)$    | $O(d)$      | $O\!\bigl((L_g + L_s \tfrac{W}{N}) d\bigr)$ |
| Inference memory        | $O(N h d)$  | $O(h d^2)$  | $O(L_g h d^2 + L_s h W d)$ |
| Inference FLOPs / step  | $O(N h d)$  | $O(h d^2)$  | $O(L_g h d^2 + L_s h W d)$ |

For $N \gg W$ the HydraLM cost collapses to the DeltaNet cost plus a
small constant. This is the regime the library is optimized for; see
`docs/benchmarks.md` for measurements.

## 7. Relation to published work

The ingredients in HydraLM are not novel in isolation:

- **Linear attention**: Katharopoulos et al., 2020.
- **Delta rule for linear attention**: Schlag et al., 2021.
- **Gated delta rule**: Yang, Wang, et al., 2024 (Gated DeltaNet).
- **Short convolutions before the mixer**: Poli et al., 2023 (H3);
  Glorioso et al., 2024 (Zamba).
- **Interleaving linear and softmax attention**: Ren et al., 2024
  (Samba); Lieber et al., 2024 (Jamba).
- **SwiGLU MLP**: Shazeer, 2020.
- **RMSNorm**: Zhang & Sennrich, 2019.
- **RoPE**: Su et al., 2021.
- **Muon optimizer**: Jordan et al., 2024.

HydraLM's purpose is to make these ideas reproducible together, not
to claim them individually. See `AUTHORS.md` for credit to prior art
and `CITATION.cff` for how to cite this repository.
