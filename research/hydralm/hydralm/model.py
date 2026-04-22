"""
HydraLM — top-level language model.

Embedding -> N x HydraBlock -> final RMSNorm -> tied LM head.

The model exposes TWO forward paths:

    forward(input_ids)                   — training / prefill, parallel over tokens
    step(input_ids, cache)               — O(1) single-token decoding

``cache`` is a list with one element per layer:
    - DeltaNet layer : {"conv_cache": (B, 3*inner, K-1), "S": (B, H, D, D)}
    - SWA layer      : {"k_cache": (B, H, W, D), "v_cache": (B, H, W, D), "pos": int}

Initialisation
--------------
Output projections (``o_proj``, ``w_down``, and the LM head when not tied)
are scaled by ``1 / sqrt(2 * n_layers)`` following the GPT-2 paper (§2.3)
and the Megatron / GPT-NeoX convention.  Without this, the residual-stream
variance grows as O(n_layers) at initialisation, which slows convergence
for deep models.  The factor of 2 accounts for the two residual
contributions per block (mixer + MLP).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import HydraConfig
from .modules import HydraBlock, RMSNorm


class HydraLM(nn.Module):
    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.blocks = nn.ModuleList(HydraBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        if cfg.tie_embeddings:
            self.lm_head = None          # will reuse embedding weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Depth-scaled re-initialisation of output projections.
        # Each residual sub-layer contributes one projection; with 2 per block
        # (mixer o_proj and MLP w_down), the fan-in scale is sqrt(2*n_layers).
        depth_std = cfg.initializer_range / math.sqrt(2.0 * cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and (
                name.endswith(".o_proj") or name.endswith(".w_down")
            ):
                nn.init.normal_(module.weight, mean=0.0, std=depth_std)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=depth_std)

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    def _project_to_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            return torch.nn.functional.linear(h, self.embed.weight)
        return self.lm_head(h)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        state: list[dict] | None = None,
        return_state: bool = False,
    ) -> dict:
        """
        Args:
            input_ids   : (B, N)
            state       : optional list of per-layer caches for streaming prefill
            return_state: if True, final state list is returned alongside logits

        Returns:
            {"logits": (B, N, vocab), "state": list | None}
        """
        x = self.embed(input_ids)
        new_states: list[dict] = []
        for i, block in enumerate(self.blocks):
            s_in = state[i] if state is not None else None
            x, s_out = block(x, s_in)
            new_states.append(s_out)
        x = self.norm_f(x)
        logits = self._project_to_logits(x)
        return {"logits": logits, "state": new_states if return_state else None}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,           # (B,) — single token
        state: list[dict] | None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """O(1) single-token decode step.

        Args:
            input_ids : (B,) integer token ids
            state     : per-layer cache dicts from the previous step, or None

        Returns:
            (logits_t, new_state)
            logits_t : (B, vocab)
            new_state: updated per-layer cache list
        """
        x_t = self.embed(input_ids)
        new_states: list[dict] = []
        for i, block in enumerate(self.blocks):
            s_in = state[i] if state is not None else None
            x_t, s_out = block.step(x_t, s_in)
            new_states.append(s_out)
        x_t = self.norm_f(x_t)
        logits_t = self._project_to_logits(x_t)
        return logits_t, new_states

    # ------------------------------------------------------------------
    def num_parameters(self, exclude_embedding: bool = False) -> int:
        """Return the number of trainable parameters."""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embedding:
            n -= self.embed.weight.numel()
        return n
