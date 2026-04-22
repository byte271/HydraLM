"""
Reference baselines for apples-to-apples comparison.

All baselines mirror HydraLM's public interface:

    model(input_ids) -> {"logits": Tensor[B, N, V]}
    model.num_parameters() -> int
    model.cfg: HydraConfig

so every benchmark / eval script can swap ``HydraLM(cfg)`` for, e.g.,
``DenseTransformer(cfg)`` without code changes.

This module also exposes the closed-form FLOP / memory / dollar model
(``flops``) used by both the analytic cost script and the ``test_claims``
gating tests, so the same equations cannot drift between them.
"""
from hydralm.baselines.transformer import DenseTransformer
from hydralm.baselines import flops

__all__ = ["DenseTransformer", "flops"]
