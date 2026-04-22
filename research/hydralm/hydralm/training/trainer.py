"""
Production training scaffolding for HydraLM.

This module wraps the usual concerns — mixed precision, gradient
accumulation, gradient clipping, cosine-with-warmup LR, gradient
checkpointing, distributed data parallel — behind a single ``Trainer``
class. It is intentionally small (< 300 lines) and has no dependency on
``accelerate``, ``lightning``, or ``transformers.Trainer`` so that the
control flow is fully auditable.

Distributed training
--------------------
If the process is launched under ``torchrun``, the trainer automatically:
  * initialises the NCCL process group,
  * wraps the model in ``FullyShardedDataParallel`` (FSDP) with a sane
    default auto-wrap policy (one shard per HydraLM block),
  * uses ``DistributedSampler``-style batch splitting (the user supplies
    a sharded iterator).

For single-GPU / CPU use, FSDP is skipped and the model is used as-is.

Mixed precision
---------------
By default bfloat16 autocast is used when a CUDA device is present; this
preserves the numeric stability the delta-rule recurrence depends on
(fp16 underflows the accumulated state matrix at long sequences).

Gradient checkpointing
----------------------
Each HydraLM block can be checkpointed. Because the chunk-wise delta-rule
kernel stores O(N·D) activations vs. the Transformer's O(N²), checkpointing
gives a *much larger* memory win here than for attention — often enabling
4× longer training sequences on the same GPU.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainingConfig:
    steps: int = 10_000
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 2_000
    checkpoint_dir: str = "./checkpoints"
    mixed_precision: str = "auto"        # "auto" | "bf16" | "fp16" | "none"
    use_fsdp: bool = True                # ignored if world_size == 1
    grad_checkpoint: bool = False
    compile: bool = False                # torch.compile the model
    seed: int = 0
    log_fn: Callable[[dict], None] | None = field(default=None, repr=False)
    # --- Optimiser --------------------------------------------------------
    # "adamw" (default) or "muon" (hybrid Muon for 2D matrices + AdamW for
    # embeddings/lm_head/biases/norms). Muon is typically ~1.3x more
    # sample-efficient at matched FLOPs on LM pretraining.
    optimizer: str = "adamw"
    muon_lr: float = 5e-3
    muon_momentum: float = 0.95


def _cosine_with_warmup(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup → cosine decay to min_lr/lr fraction."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = cfg.min_lr / cfg.lr
    return floor + (1.0 - floor) * cos


def _is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def _maybe_init_distributed() -> tuple[int, int, int]:
    if not _is_distributed():
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def _pick_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "none":
        return None
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return None
    raise ValueError(f"unknown mixed_precision={mixed_precision}")


class Trainer:
    """Minimal-but-complete training loop."""

    def __init__(self, model: nn.Module, cfg: TrainingConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.rank, self.world, self.local = _maybe_init_distributed()
        self.device = (
            torch.device(f"cuda:{self.local}") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = model.to(self.device)

        if cfg.grad_checkpoint:
            self._enable_gradient_checkpointing(model)

        if cfg.use_fsdp and self.world > 1:
            model = self._wrap_fsdp(model)

        if cfg.compile:
            model = torch.compile(model, mode="default", fullgraph=False)

        self.model = model

        if cfg.optimizer == "muon":
            from hydralm.optim import build_hybrid_optimizer
            self.optimizer = build_hybrid_optimizer(
                model,
                muon_lr=cfg.muon_lr,
                muon_momentum=cfg.muon_momentum,
                muon_weight_decay=0.0,
                adamw_lr=cfg.lr,
                adamw_betas=cfg.betas,
                adamw_weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw":
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # RMSNorm params, biases, and embeddings skip weight decay.
                if p.ndim <= 1 or n.endswith(".bias") or "embed" in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
            self.optimizer = AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.lr, betas=cfg.betas, fused=torch.cuda.is_available(),
            )
        else:
            raise ValueError(f"unknown optimizer={cfg.optimizer!r}; expected 'adamw' or 'muon'")

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda s: _cosine_with_warmup(s, cfg))

        self.amp_dtype = _pick_dtype(cfg.mixed_precision)
        self.scaler = torch.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        self.step = 0

    # --- distributed / memory helpers -------------------------------------

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """Wrap each HydraBlock's forward in a checkpoint boundary."""
        from torch.utils.checkpoint import checkpoint as _ckpt
        from hydralm.modules.block import HydraBlock
        for m in model.modules():
            if isinstance(m, HydraBlock):
                orig = m.forward
                def ckpt_forward(*args, _orig=orig, **kw):
                    return _ckpt(_orig, *args, use_reentrant=False, **kw)
                m.forward = ckpt_forward  # type: ignore[assignment]

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from hydralm.modules.block import HydraBlock
        # ModuleWrapPolicy is the stable API for wrapping named module types.
        # The old lambda approach worked until PyTorch 2.1 changed the
        # auto_wrap_policy contract: the policy must now be a proper callable
        # that accepts (module, recurse, unwrapped_params) and returns bool,
        # and it must NOT return True for the root module itself.
        # ModuleWrapPolicy handles all of this correctly.
        policy = ModuleWrapPolicy({HydraBlock})
        return FSDP(
            model,
            auto_wrap_policy=policy,
            device_id=self.local,
            use_orig_params=True,
        )

    # --- main training loop -----------------------------------------------

    def fit(
        self,
        data: Iterable[tuple[Tensor, Tensor]] | Iterator[tuple[Tensor, Tensor]],
        on_eval: Callable[[nn.Module, int], dict] | None = None,
    ) -> None:
        cfg = self.cfg
        data_iter = iter(data)
        t0 = time.time()

        while self.step < cfg.steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=self.device)
            for _ in range(cfg.grad_accum):
                try:
                    ids, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(data)
                    ids, labels = next(data_iter)
                ids = ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.amp_dtype is not None:
                    ctx = torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                else:
                    ctx = torch.enable_grad()
                with ctx:
                    out = self.model(ids)
                    logits = out["logits"] if isinstance(out, dict) else out
                    # causal LM next-token loss
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    ) / cfg.grad_accum

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += loss.detach()

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.rank == 0 and self.step % cfg.log_every == 0:
                dt = time.time() - t0
                rec = {
                    "step": self.step,
                    "loss": float(loss_accum.item()),
                    "lr": self.scheduler.get_last_lr()[0],
                    "tok_per_s": cfg.batch_size * cfg.grad_accum * ids.size(1) * cfg.log_every / max(dt, 1e-9),
                }
                (cfg.log_fn or (lambda r: print(f"[train] {r}", flush=True)))(rec)
                t0 = time.time()

            if on_eval is not None and self.step % cfg.eval_every == 0 and self.rank == 0:
                on_eval(self.model, self.step)

            if self.step % cfg.save_every == 0 and self.rank == 0:
                self._save(f"step_{self.step}")

        if self.world > 1:
            dist.barrier()
            if dist.is_initialized():
                dist.destroy_process_group()

    def _save(self, tag: str) -> None:
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"{tag}.pt")
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
        }
        torch.save(state, path)


__all__ = ["TrainingConfig", "Trainer"]
