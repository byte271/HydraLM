# Training Guide

This document covers the training APIs that exist in the current HydraLM
codebase: the minimal `train_tiny.py` script, the built-in `Trainer`, mixed
precision, optimizer choices, FSDP integration, checkpointing, and the
optional MTP loss path.

## Quick start: `train_tiny.py`

The fastest way to exercise the full training stack is the character-level
demo script:

```bash
cd research/hydralm

python scripts/train_tiny.py \
    --data data/tinyshakespeare.txt \
    --steps 2000 \
    --batch-size 16 \
    --seq-len 512 \
    --d-model 256 \
    --n-layers 6 \
    --n-heads 4 \
    --optimizer adamw
```

If you want the hybrid optimizer path instead, switch to:

```bash
python scripts/train_tiny.py \
    --data data/tinyshakespeare.txt \
    --optimizer muon \
    --muon-lr 5e-3
```

The script instantiates `HydraConfig`, builds `HydraLM`, chooses either
plain AdamW or the hybrid Muon optimizer, and runs a compact training loop
with sampling checkpoints printed to stdout.

## Built-in trainer

The reference trainer lives in `hydralm/training/trainer.py` and is driven by
`TrainingConfig`, not `TrainerConfig`.

```python
from hydralm import HydraConfig, HydraLM
from hydralm.training.trainer import TrainingConfig, Trainer

cfg = HydraConfig(
    vocab_size=32_000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    swa_window=512,
)
model = HydraLM(cfg)

tcfg = TrainingConfig(
    steps=10_000,
    batch_size=8,
    grad_accum=1,
    lr=3e-4,
    min_lr=3e-5,
    warmup_steps=500,
    weight_decay=0.1,
    mixed_precision="auto",
    optimizer="adamw",
    compile=False,
)

trainer = Trainer(model, tcfg)
trainer.fit(data_iterable)
```

What the trainer currently provides:

- gradient accumulation
- cosine decay with warmup
- gradient clipping
- optional `torch.compile`
- optional gradient checkpointing
- optional FSDP when launched under distributed execution
- periodic logging, evaluation callbacks, and checkpoint saves

## Optimizer options

`TrainingConfig.optimizer` accepts:

- `"adamw"` — the default trainer path
- `"muon"` — hybrid Muon for 2-D matrices plus AdamW for the rest

If you want to construct the hybrid optimizer directly:

```python
from hydralm.optim import build_hybrid_optimizer

optim = build_hybrid_optimizer(
    model,
    muon_lr=5e-3,
    muon_momentum=0.95,
    adamw_lr=3e-4,
    adamw_betas=(0.9, 0.95),
    adamw_weight_decay=0.1,
)
```

The trainer uses the same helper internally when `optimizer="muon"`.

## Mixed precision

`TrainingConfig.mixed_precision` supports:

- `"auto"`
- `"bf16"`
- `"fp16"`
- `"none"`

`"auto"` picks bf16 when supported on CUDA, falls back to fp16 on CUDA
otherwise, and disables autocast on CPU.

Because the delta-rule recurrence is sensitive to low-precision accumulation,
bf16 is the safer mixed-precision option when your hardware supports it.

## Learning-rate schedule

The trainer uses a linear warmup followed by cosine decay:

- warmup from `0` to `lr` over `warmup_steps`
- cosine decay from `lr` to `min_lr` over the remaining training steps

The relevant fields are:

- `lr`
- `min_lr`
- `warmup_steps`
- `steps`

## Distributed training and FSDP

The trainer checks the distributed environment automatically. When launched
under `torchrun` and `use_fsdp=True`, it wraps the model in
`FullyShardedDataParallel`.

That means the built-in training path is:

- single-process / single-device when no distributed environment is present
- FSDP-backed when the process group is available and `use_fsdp=True`

Gradient checkpointing is separate and controlled by `grad_checkpoint`.

## Checkpointing

The built-in trainer saves `.pt` checkpoints containing:

- `step`
- `model`
- `optim`
- `sched`

The save path is controlled by:

- `checkpoint_dir`
- `save_every`

Reloading a native checkpoint looks like this:

```python
import torch
from hydralm import HydraConfig, HydraLM

cfg = HydraConfig(...)
model = HydraLM(cfg)

state = torch.load("checkpoints/step_2000.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state["model"])
```

The trainer checkpoint does not currently persist `HydraConfig`, so keep your
config alongside the checkpoint in your own training workflow.

## MTP auxiliary loss

When `cfg.mtp_depth > 0`, the model can emit an auxiliary next-k loss during
training:

```python
import torch.nn.functional as F

out = model(input_ids, compute_mtp=True)
loss = F.cross_entropy(
    out["logits"][:, :-1].reshape(-1, cfg.vocab_size),
    input_ids[:, 1:].reshape(-1),
    ignore_index=-100,
)
if out["mtp_aux_loss"] is not None:
    loss = loss + out["mtp_aux_loss"]
```

`out["mtp_aux_loss"]` is already scaled by `cfg.mtp_loss_weight`.

## Retrieval-aware training notes

Retrieval layers do not require a separate training loop. Once scheduled via
`retrieval_every` or `layer_types`, they train like any other block in the
backbone.

Two practical knobs matter most:

1. `retrieval_chunk_size` controls routing granularity.
2. `retrieval_top_k` controls how much distant context each query chunk sees.

## Common failure modes

- If you see NaNs, start by checking your precision mode.
- If long runs are too memory-heavy, try `grad_checkpoint=True`.
- If you want the hybrid optimizer, make sure `optimizer="muon"` is set;
  plain AdamW is the default.
- If you need exact trainer field names, follow `TrainingConfig` in
  `hydralm/training/trainer.py` rather than older examples.

See `docs/api.md` for signatures, `docs/deployment.md` for inference-time
usage, and `docs/faq.md` for troubleshooting.
