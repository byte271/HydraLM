# Fact Bank

The **fact bank** is HydraLM's optional *non-parametric memory*: an
external key-value store that is queried at every generation step
and whose top-k retrieved values are mixed into the final hidden
state before the LM head.

Unlike the DeltaNet recurrent state — which is an *implicit*, lossy,
capacity-bounded memory — the fact bank is *explicit*, lossless, and
capacity-unbounded. The two memories complement each other:

- The DeltaNet state remembers *patterns* from recent tokens.
- The fact bank remembers *facts* that were deliberately written.

The implementation lives in `hydralm/memory/fact_bank.py`.

## When to use it

Use the fact bank when you need:

- **Large, updatable long-term knowledge** that must not be
  overwritten by new tokens (names, IDs, timestamps, documents).
- **Online learning without retraining** — write new facts between
  prompts and they will influence the next generation immediately.
- **Explainability** — every retrieved fact is a specific `(key,
  value)` pair you chose, so you can show *why* the model said what
  it said.

Do *not* use the fact bank as a replacement for attention inside the
model. It is not a first-class differentiable memory; gradients do
not flow through it by default.

## API at a glance

```python
from hydralm.memory.fact_bank import FactBank

bank = FactBank(dim=512, capacity=10_000, device="cuda")

# Writing: associate a normalized key with a value vector.
bank.write(key, value)        # key, value: (dim,) or (B, dim)

# Querying: retrieve top-k nearest keys (cosine similarity).
values, scores = bank.query(query, k=8)
# values: (B, k, dim), scores: (B, k)

# Snapshotting.
state = bank.state_dict()
bank.load_state_dict(state)
```

See `docs/api.md` for the full parameter list and the exact return
shapes.

## How retrieval is mixed into the model

`hydralm.model.HydraLM.forward` accepts an optional `fact_bank`
argument:

```python
model = HydraLM(cfg).cuda()
bank  = FactBank(dim=cfg.d_model, capacity=10_000)

# Populate the bank with documents from your corpus.
for text, ids in corpus:
    h   = model.embed(ids)              # (1, T, d)
    key = h.mean(dim=1).squeeze(0)       # (d,)
    bank.write(key, h.squeeze(0)[-1])   # store last hidden state

# Use the bank at inference.
out = model(ids, fact_bank=bank, fact_topk=8)
```

At every *last* layer of the stack, the final hidden state `h_last`
is used as a query. The top-k retrieved values are aggregated with
softmax-weighted scores and summed into `h_last` through a small
learned gate (initialized to zero, so the fact bank has no effect
until it is trained or configured to be active).

The gate is a single scalar per layer; the mixing is therefore
explicitly *additive*, not replacing the model's own prediction.

## Scaling and quantization

- **Exact search** is used by default. For banks up to ~1M entries on
  a single GPU this is fine and much simpler than ANN indices.
- **FP16 keys** halve memory at a negligible recall cost. Set
  `bank.dtype = torch.float16` at construction.
- For banks beyond GPU memory, use the `to_cpu()` / `to_disk()`
  helpers documented in `docs/api.md`. Query latency grows
  proportionally; consider a FAISS or ScaNN backend for production.

## Relation to retrieval-augmented generation

The fact bank is deliberately *not* RAG. RAG concatenates retrieved
text into the prompt; the fact bank injects retrieved *activations*
directly into the residual stream. This means:

- **Pro**: no context-length budget is consumed per retrieval.
- **Pro**: keys and values can be anything — they don't have to be
  decodable text.
- **Con**: retrieved activations are only as useful as the embedding
  space the model inhabits, so the fact bank needs to be populated
  with activations from the *same* model that will query it.

You can combine the two freely: run a RAG pipeline over the prompt
text and populate the fact bank with the retrieved documents'
activations. See `scripts/online_learning_demo.py` for a worked
example.

## Evaluation

`hydralm.eval.online_learning` contains a small, deterministic
benchmark that measures the fact-bank's write/read round-trip
accuracy. Expected behaviour:

- **Retention**: >95% exact-match recall at `capacity = 10_000`
  immediately after writing.
- **Interference**: approximately log-linear with the write rate — at
  1k writes/s, `p@1` degrades by <5 percentage points per hour.

Reproduce with:

```bash
python scripts/online_learning_demo.py --capacity 10000 --writes 1000000
```

See `docs/evaluation.md` for the full reproduction protocol and
`tests/test_fact_bank.py` for the unit-test contract.
