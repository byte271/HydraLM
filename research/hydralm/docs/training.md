# Training Guide

This document covers how to train a HydraLM from scratch or fine-tune
a checkpoint. The reference implementation is in
`hydralm/training/trainer.py` and the minimal demo script is
`scripts/train_tiny.py`.

## Quick start: train a tiny model on TinyShakespeare

```bash
# From the repository root.
pip install -e .[dev]
python scripts/train_tiny.py \
    --data data/tinyshakespeare.txt \
    --d-model 256 --n-layers 6 --n-heads 4 \
    --seq-len 1024 --batch-size 16 \
    --steps 2000 --lr 3e-4 \
    --save checkpoints/tiny
```

On a single A100 this reaches ~1.7 bits per byte (≈2.5 val loss in
nats) in under 10 minutes. Checkpoints are written with
`HydraLM.save_pretrained`; reload them with `HydraLM.from_pretrained`.

## The training loop

The reference trainer is intentionally minimal — around 200 lines —
and is meant to be copied and modified rather than configured to
death.

```python
import torch
from hydralm import HydraLM, HydraConfig
from hydralm.training.trainer import Trainer, TrainerConfig

cfg = HydraConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
    seq_len=2048, swa_window=256,
)
model = HydraLM(cfg).cuda()

tcfg = TrainerConfig(
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=1_000,
    total_steps=50_000,
    precision="bfloat16",
    compile=True,
)
trainer = Trainer(model, tcfg)

for step, (inp, tgt) in enumerate(loader):
    loss = trainer.step(inp.cuda(), tgt.cuda())
    if step % 100 == 0:
        print(f"step {step:>6}  loss {loss:.4f}")
```

## Optimizer: Muon + AdamW

HydraLM uses the hybrid **Muon + AdamW** optimizer
(`hydralm.optim.HybridMuonAdamW`) by default. It applies:

- **Muon** (Jordan et al., 2024) to the 2-D matrix parameters of
  every linear layer inside each block — this is where Muon's
  orthogonalization step is helpful.
- **AdamW** to everything else: embeddings, LM head, RMSNorm
  weights, biases, and any 1-D or 0-D parameter.

Muon typically yields a 1.2–1.6× wall-clock speedup at matched loss
versus AdamW-only. If your hardware does not support the Newton-Schulz
iteration efficiently (older GPUs, CPU), pass `optimizer="adamw"` to
`TrainerConfig` to fall back to standard AdamW.

## Precision

Three precision modes are supported and selected with
`TrainerConfig.precision`:

| Mode         | Weights  | Compute           | Grad state |
| ------------ | -------- | ----------------- | ---------- |
| `"float32"`  | fp32     | fp32              | fp32       |
| `"bfloat16"` | fp32     | bfloat16 autocast | fp32       |
| `"fp16"`     | fp32     | fp16 + GradScaler | fp32       |

**`bfloat16` is the recommended default** on Ampere+. HydraLM is
numerically insensitive to bf16 in the compute path because the
delta-rule accumulator `S` is internally upcast to fp32 (see
`docs/theory.md` §4).

## Learning-rate schedule

The default schedule is a cosine decay with linear warmup, defined in
`TrainerConfig`:

- Warmup: `0 → lr` linearly over `warmup_steps`.
- Decay:  `lr → lr * min_lr_ratio` as cosine over the remaining
  `total_steps - warmup_steps`.

`min_lr_ratio` defaults to `0.1`, matching the Chinchilla protocol.
For small runs (`total_steps < 5_000`) we recommend a constant LR
after warmup.

## Gradient clipping and weight decay

- `grad_clip=1.0` is always on.
- Weight decay is **not** applied to:
  - embeddings (`embed_tokens.weight`),
  - LM head (tied to embeddings anyway),
  - any `RMSNorm` weight,
  - any bias,
  - any gate/scaling scalar inside `GatedDeltaNet`.

The trainer does the bucketing for you; see
`Trainer._build_param_groups`.

## Distributed training

The reference trainer supports single-node DDP out of the box:

```bash
torchrun --nproc-per-node=8 scripts/train_tiny.py \
    --d-model 1024 --n-layers 24 --batch-size 8 ...
```

The `Trainer` will detect DDP automatically via
`torch.distributed.is_initialized()` and wrap the model with
`DistributedDataParallel` using `gradient_as_bucket_view=True`.

FSDP is *not* built in — HydraLM's target scale is sub-billion
parameters, which comfortably fits in a single GPU's memory under
bf16. If you need FSDP you should use your own training framework
(Lightning, `nanotron`, `torchtitan`) and simply import `HydraLM` and
`HybridMuonAdamW` from this package.

## Checkpointing

```python
# Save.
model.save_pretrained("checkpoints/step-10000")
# -> creates config.json + pytorch_model.bin

# Load.
from hydralm import HydraLM
model = HydraLM.from_pretrained("checkpoints/step-10000").cuda()
```

See `docs/api.md` for the full serialization format, and
`hydralm/deploy/hf_adapter.py` for how to export the same checkpoint
as a Hugging Face transformers model.

## Common pitfalls

- **NaNs after a few hundred steps** — almost always means the
  delta-rule accumulator is being run in fp16 somewhere. Check that
  you are using `precision="bfloat16"` or `"float32"`, not `"fp16"`
  without a scaler, and that any `@torch.autocast` context you add
  does not include `dtype=torch.float16`.

- **Loss plateaus at ~`log(vocab_size)`** — the LM head is not tied
  to the embeddings, or your tokenizer mismatches the vocab_size in
  `HydraConfig`. Verify `cfg.vocab_size == tokenizer.vocab_size`.

- **Slow training** — make sure `compile=True` in `TrainerConfig`
  and that you are on PyTorch ≥ 2.4. The first iteration is slow
  (compile) but every subsequent iteration should hit ~85% MFU on an
  A100 at `seq_len=2048`.

See `docs/faq.md` for more troubleshooting.
