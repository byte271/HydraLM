"""
Test-time associative memory built on the delta rule.

The gated DeltaNet update
    S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T
is bit-for-bit Widrow--Hoff online least-squares on the loss
    L(S) = 1/2 ||S k_t - v_t||^2.

So the recurrent state is not just "sequence compression" -- it is a
*live least-squares solver*. A single forward pass over (k_i, v_i)
pairs with alpha=1, beta=1, and unit-norm keys yields an S such that
S k_j ~= v_j for every written pair, with retrieval error scaling as
O(sqrt(N/d)) for random keys.

This module exposes that capability as a first-class API
(`FactBank`), so users can treat the linear layer as an in-context
memory store: write facts at inference time, read them back, overwrite
them, all with zero gradient updates and O(H * d^2) total memory
independent of how many facts are written.
"""
from hydralm.memory.fact_bank import FactBank, FactBankStats

__all__ = ["FactBank", "FactBankStats"]
