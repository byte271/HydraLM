# Evaluation

HydraLM ships a small, deterministic evaluation harness under
`hydralm/eval/`. Its purpose is to make the claims in `RESULTS.md`
reproducible on a single GPU; it is *not* a replacement for
general-purpose evaluation stacks like `lm-evaluation-harness`.

Every evaluation supports a `--seed` flag, uses a frozen dataset
splits or a deterministic synthetic generator, and writes a JSON
report that is diff-friendly against the committed `results.json`.

## The four core benchmarks

| Module                                  | What it measures                              | Runtime (A100) |
| --------------------------------------- | --------------------------------------------- | -------------- |
| `hydralm.eval.mqar`                     | Multi-Query Associative Recall                | ~3 min         |
| `hydralm.eval.long_context`             | Needle-in-a-Haystack over 1M tokens           | ~25 min        |
| `hydralm.eval.online_learning`          | Fact bank write/read round-trip accuracy      | ~2 min         |
| `hydralm.eval.claims`                   | Composite pass/fail against `docs/claims.md`  | ~30 min        |

They are exposed as both library functions and CLI scripts in
`scripts/`.

## 1. Multi-Query Associative Recall (MQAR)

MQAR is the canonical stress test for linear-attention layers. A
sequence is a concatenation of `(key, value)` pairs followed by a
query that asks for a specific stored key. The model must learn to
retrieve the correct value despite never seeing the key-value pair
twice.

```bash
python scripts/run_mqar.py \
    --model checkpoints/step-50000 \
    --n-pairs 64 128 256 512 1024 \
    --seq-len 8192
```

The report records exact-match accuracy at each `n_pairs`. HydraLM at
160M with the default 4:1 schedule typically maintains >95% accuracy
at `n_pairs <= 512` and degrades gracefully beyond.

## 2. Long-context (Needle-in-a-Haystack)

The long-context benchmark places a single "needle" fact at a
randomized depth inside a long distractor corpus and asks the model to
retrieve it.

```bash
python scripts/needle_in_haystack.py \
    --model checkpoints/step-50000 \
    --max-length 1048576 \
    --depths 0.1 0.25 0.5 0.75 0.9
```

The report records retrieval accuracy for each `(depth, length)` pair
and writes a heatmap to `results/needle.png`.

The default configuration probes lengths `{4k, 16k, 64k, 256k, 1M}`
and depths `{0.1, 0.25, 0.5, 0.75, 0.9}`. Expected behaviour at 160M:

- **Within SWA window** (last `W=256` tokens): ~100% accuracy at all
  depths.
- **Beyond SWA window**: accuracy falls to the DeltaNet state's
  capacity bound, which in practice is around 70–80% at 1M tokens
  when the needle is a single short fact.

See `docs/benchmarks.md` for full numbers.

## 3. Fact-bank online learning

```bash
python scripts/online_learning_demo.py \
    --capacity 10000 --writes 1000000
```

Write 1M random `(key, value)` pairs, query a held-out 10% subset,
and report `p@1` and `p@8` retrieval accuracy. The pass threshold in
`docs/claims.md` is `p@1 >= 0.95`.

## 4. Claims benchmark

The claims benchmark is the one you should run **before cutting a
release**. It chains the three benchmarks above plus a few
correctness checks (equivalence of training-time and inference-time
forwards, shape sanity, LR schedule shape) and produces a single
pass/fail report:

```bash
python scripts/reproduce_claims.py --model checkpoints/step-50000
```

Output:

```
[PASS] training/inference equivalence    max |delta| = 2.1e-6
[PASS] shapes sanity                     all tensors finite, no leaks
[PASS] MQAR p@1 @ n_pairs=256            0.982   >= 0.95
[PASS] needle depth=0.5 len=256k         0.94    >= 0.90
[PASS] fact-bank p@1                     0.987   >= 0.95
```

Failures are fatal: CI in `byte271/hydralm` gates releases on this
script.

## 5. Writing a new evaluation

A new evaluation should be a single module under `hydralm/eval/`
that:

1. Exposes one top-level function `run(model, *, seed=0, **cfg)
   -> dict` whose return value is a JSON-serializable report.
2. Uses deterministic inputs — either a frozen dataset file under
   `data/` or a synthetic generator seeded from `seed`.
3. Writes *no* global state (no cached datasets, no `torch.manual_seed`
   leaks); use `torch.Generator` objects explicitly.

Register the new evaluation in `hydralm/eval/__init__.py` so it is
picked up by `scripts/reproduce_claims.py` if you want it gated in
CI.

See `hydralm/eval/mqar.py` for the smallest complete example.
