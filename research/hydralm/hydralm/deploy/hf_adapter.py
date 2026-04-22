"""
HuggingFace-compatible adapter for HydraLM.

Exposes a ``HydraLMForCausalLM`` whose public surface is a strict subset of
``transformers.GPT2LMHeadModel``. Any existing training or serving code
that does

    model = AutoModelForCausalLM.from_pretrained(...)
    out = model(input_ids=ids, labels=ids)
    loss, logits = out.loss, out.logits

continues to work when the model is swapped for ``HydraLMForCausalLM`` —
no changes to the surrounding pipeline required. This is the literal
meaning of "drop-in replacement for the Transformer".

Design notes
------------
We deliberately DO NOT inherit from ``transformers.PreTrainedModel``:
  1. It would force a hard dependency on ``transformers`` for the whole
     project (we want the core ``hydralm`` package to be importable with
     only ``torch`` installed).
  2. It would force HydraLM through the KV-cache abstraction, which is
     the exact quadratic bottleneck we're replacing.

Instead we provide a tiny ``ModelOutput``-compatible return type and a
``generate()`` method that uses HydraLM's own O(1) recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hydralm.config import HydraConfig
from hydralm.model import HydraLM


@dataclass
class CausalLMOutput:
    """Mimics ``transformers.modeling_outputs.CausalLMOutputWithPast``."""
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    past_key_values: Optional[list] = None  # HydraLM recurrent state, not KV cache


class HydraLMForCausalLM(nn.Module):
    """Drop-in replacement for ``GPT2LMHeadModel`` / ``LlamaForCausalLM``."""

    def __init__(self, config: HydraConfig):
        super().__init__()
        self.config = config
        self.model = HydraLM(config)

    # --- GPT-2 API parity -------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed

    def set_input_embeddings(self, new_embed: nn.Module) -> None:
        self.model.embed = new_embed

    def get_output_embeddings(self) -> nn.Module:
        """Return the LM head. If embeddings are tied, returns the shared
        embedding weight wrapped in a ``Linear`` view."""
        if self.model.lm_head is None:
            # synthesise a view that shares weight with the embedding
            e = self.model.embed
            head = nn.Linear(e.embedding_dim, e.num_embeddings, bias=False)
            head.weight = e.weight
            return head
        return self.model.lm_head

    def tie_weights(self) -> None:
        # HydraLM ties via `lm_head is None`; nothing to do.
        self.config.tie_embeddings = True
        self.model.lm_head = None

    def resize_token_embeddings(self, new_size: int) -> nn.Module:
        old = self.model.embed
        new = nn.Embedding(new_size, old.embedding_dim,
                           padding_idx=getattr(old, "padding_idx", None)).to(old.weight.device)
        with torch.no_grad():
            n = min(new_size, old.num_embeddings)
            new.weight[:n].copy_(old.weight[:n])
        self.model.embed = new
        if self.model.lm_head is not None:
            self.model.lm_head = nn.Linear(old.embedding_dim, new_size, bias=False).to(old.weight.device)
        self.config.vocab_size = new_size
        return new

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,   # accepted & ignored (causal only)
        past_key_values: Optional[list] = None,    # HydraLM state (not KV cache)
        use_cache: bool = False,                   # kept for API compatibility
        return_dict: bool = True,
        **_unused,
    ) -> CausalLMOutput:
        out = self.model(
            input_ids,
            state=past_key_values,
            return_state=bool(use_cache),
        )
        logits = out["logits"]
        state = out.get("state")

        loss: Optional[Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=state if use_cache else None,
        )

    # --- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> Tensor:
        """HydraLM-native generate — uses the O(1) recurrent step, not a KV cache.

        ``do_sample=False`` is mapped to temperature=0 (greedy argmax)."""
        from hydralm.generation import generate
        return generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    # --- serialization ----------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        import json, os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "hydra_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, directory: str, **overrides) -> "HydraLMForCausalLM":
        """Re-load a model previously saved with :meth:`save_pretrained`.

        The block schedule (``layer_types``) is preserved exactly as it
        was at save time. Dropping it here — as earlier versions of the
        adapter did — silently reverted custom schedules to the default
        ``swa_every``-computed schedule and caused ``load_state_dict`` to
        fail whenever the checkpoint was trained with a non-default
        layout.
        """
        import json, os
        with open(os.path.join(directory, "hydra_config.json")) as f:
            raw = json.load(f)
        # ``HydraConfig.__post_init__`` only auto-derives ``layer_types``
        # when it is ``None``; passing the saved tuple/list through is
        # correct and required for checkpoints with a custom schedule.
        cfg = HydraConfig(**{**raw, **overrides})
        model = cls(cfg)
        state = torch.load(
            os.path.join(directory, "pytorch_model.bin"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state)
        return model


__all__ = ["HydraLMForCausalLM", "CausalLMOutput"]
