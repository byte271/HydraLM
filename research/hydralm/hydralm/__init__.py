"""HydraLM — a sub-quadratic language model."""
from .config import HydraConfig
from .generation import generate
from .model import HydraLM
from .spec_decoding import speculative_generate, SpecDecodingStats
from .optim import Muon, HybridMuonAdamW, build_hybrid_optimizer

__version__ = "0.2.0"

__all__ = [
    "HydraConfig",
    "HydraLM",
    "generate",
    "speculative_generate",
    "SpecDecodingStats",
    "Muon",
    "HybridMuonAdamW",
    "build_hybrid_optimizer",
]
