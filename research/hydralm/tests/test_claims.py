"""
Numerical gates for the five headline research claims.

The check logic itself lives in ``hydralm.eval.claims`` so that the test
suite and ``scripts/reproduce_claims.py`` cannot drift apart. This file is
intentionally a thin wrapper: each test runs the corresponding checker at
its CPU-cheap defaults and asserts ``result.passed``, surfacing the
measured numbers on failure so CI logs stay self-diagnosing.

Claims
------
1. Linear Complexity O(N)
2. Lossless Accuracy (~ Transformer) on recall (MQAR)
3. Support for 1M-10M tokens (constant-state streaming)
4. 90% Cost Reduction vs dense Transformer at long context
5. Drop-in Replacement for Transformers (API + param-matched + HF adapter)
6. Zero-Gradient Test-Time Learning (FactBank online associative memory)
"""
from __future__ import annotations

from hydralm.eval.claims import (
    check_claim_1_linear_complexity,
    check_claim_2_lossless_mqar,
    check_claim_3_constant_state,
    check_claim_4_cost_reduction,
    check_claim_5_drop_in,
    check_claim_6_online_learning,
)


def _assert_passed(result) -> None:
    assert result.passed, (
        f"{result.name} FAILED\n"
        f"  thresholds: {result.thresholds}\n"
        f"  measured  : {result.measured}\n"
        f"  notes     : {result.notes}"
    )


def test_claim_1_linear_complexity():
    """HydraLM FLOPs must fit O(N^1); Transformer must be super-linear."""
    _assert_passed(check_claim_1_linear_complexity())


def test_claim_2_lossless_accuracy_on_mqar():
    """HydraLM (hybrid) reaches >=90% of Transformer recall on MQAR.

    Cheap gate: 2 KV pairs / 1 query / seq_len=32. The point of *this*
    gate is not to stress-test recall -- it is to ensure HydraLM does not
    collapse relative to a Transformer on a recall task at all. The heavy
    stress test (D=8, Q=8, long budget) lives in
    ``scripts/reproduce_claims.py`` and is reported in ``RESULTS.md``.
    """
    _assert_passed(check_claim_2_lossless_mqar())


def test_claim_3_million_token_streaming_is_constant_state():
    """Streaming state memory must be O(1): bounded regardless of N,
    and a live 1K prefill + 128 single-token steps must stay finite."""
    _assert_passed(check_claim_3_constant_state())


def test_claim_4_ninety_percent_cost_reduction():
    """At 128K tokens, HydraLM saves >=90% FLOPs AND >=90% memory, and
    the savings are monotone in N."""
    _assert_passed(check_claim_4_cost_reduction())


def test_claim_5_drop_in_transformer_replacement():
    """HydraLM and DenseTransformer share the API and param budget, and
    the HF adapter's ``generate`` works end-to-end."""
    _assert_passed(check_claim_5_drop_in())


def test_claim_6_zero_gradient_online_learning():
    """The delta-rule state is an online associative memory strong
    enough to replace a KV cache for key-value retrieval, with
    destructive overwrite, no gradients, and a state whose size is
    independent of the number of facts written. Cheap gate:
    d=128, H=2, N=64 -- strictly within LMS capacity."""
    _assert_passed(check_claim_6_online_learning())
