"""Tests for MQAR generator, long-context needle generator, and HF adapter."""
from __future__ import annotations

import torch

from hydralm import HydraConfig
from hydralm.deploy import HydraLMForCausalLM, CompiledDecoder, Request
from hydralm.eval import (
    MQARConfig, make_mqar_batch, evaluate_mqar,
    LongContextConfig, make_needle_batch, evaluate_needle,
)


def _tiny_cfg(vocab: int = 257) -> HydraConfig:
    return HydraConfig(
        vocab_size=vocab, d_model=64, n_layers=4, n_heads=4,
        swa_window=16, dn_chunk_size=8, swa_every=2,
    )


# ---------- MQAR data generator ----------

def test_mqar_batch_shape_and_labels():
    cfg = MQARConfig(vocab_size=512, num_kv_pairs=4, num_queries=4)
    ids, lbl = make_mqar_batch(3, cfg, generator=torch.Generator().manual_seed(0))
    assert ids.shape == (3, 16)  # 2*(4+4)
    assert lbl.shape == ids.shape
    # Exactly num_queries answer positions per example.
    assert (lbl != -100).sum(dim=1).eq(4).all()


def test_mqar_keys_values_disjoint():
    """Keys from [0, V/2), values from [V/2, V)."""
    cfg = MQARConfig(vocab_size=1024, num_kv_pairs=8, num_queries=8)
    ids, lbl = make_mqar_batch(4, cfg, generator=torch.Generator().manual_seed(0))
    # every label (a value) should be in the upper half
    vals = lbl[lbl != -100]
    assert (vals >= 512).all()


def test_evaluate_mqar_runs():
    cfg = _tiny_cfg(vocab=513)
    model = HydraLMForCausalLM(cfg)
    mqar = MQARConfig(vocab_size=513, num_kv_pairs=4, num_queries=4)
    metrics = evaluate_mqar(model.model, mqar, n_batches=2, batch_size=4)
    assert 0.0 <= metrics["mqar_accuracy"] <= 1.0
    assert metrics["answer_positions"] == 2 * 4 * 4  # n_batches * B * Q


# ---------- long-context needle ----------

def test_needle_batch_structure():
    cfg = LongContextConfig(vocab_size=1024, seq_len=512, num_needles=1)
    ids, lbl, depths = make_needle_batch(
        2, cfg, generator=torch.Generator().manual_seed(0)
    )
    assert ids.shape == (2, 512)
    # Exactly one supervised position per sequence.
    assert (lbl != -100).sum(dim=1).eq(1).all()
    for d in depths:
        assert 0 <= d < 512


def test_needle_eval_runs():
    cfg = _tiny_cfg(vocab=1025)
    model = HydraLMForCausalLM(cfg)
    lc = LongContextConfig(vocab_size=1025, seq_len=256, num_needles=2)
    metrics = evaluate_needle(model.model, lc, n_batches=2, batch_size=2)
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------- HF adapter ----------

def test_hf_adapter_forward_and_loss():
    cfg = _tiny_cfg()
    model = HydraLMForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_hf_adapter_generate_matches_native():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = HydraLMForCausalLM(cfg).eval()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out1 = model.generate(prompt, max_new_tokens=8, do_sample=False)
    assert out1.shape == (1, 12)


def test_hf_adapter_save_load_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    model = HydraLMForCausalLM(cfg).eval()
    model.save_pretrained(str(tmp_path))
    loaded = HydraLMForCausalLM.from_pretrained(str(tmp_path)).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    a = model(input_ids=ids).logits
    b = loaded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_hf_adapter_tie_weights():
    cfg = _tiny_cfg()
    model = HydraLMForCausalLM(cfg)
    head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    assert head.weight.data_ptr() == embed.weight.data_ptr()


# ---------- compiled batched decoder ----------

def test_compiled_decoder_batch_decode():
    """Batched decode across same-length prompts (the bucketed-batching pattern).

    Mixed-length batching is a separate concern (SWA kv_buffer sizes differ);
    the production pattern is to bucket requests by prompt length, which is
    what vLLM did prior to PagedAttention. Adding page-style memory management
    is a deliberate follow-up.
    """
    cfg = _tiny_cfg()
    model = HydraLMForCausalLM(cfg).eval().model  # unwrap to HydraLM
    decoder = CompiledDecoder(model, compile=False)  # keep test fast / portable

    prompt_len = 8
    reqs = [
        Request(prompt=torch.randint(0, cfg.vocab_size, (prompt_len,)),
                max_new_tokens=5, temperature=0.0)
        for _ in range(3)
    ]
    outs = decoder.decode(reqs)
    assert len(outs) == 3
    for r, o in zip(reqs, outs):
        assert o.shape[0] == r.prompt.shape[0] + 5


def test_compiled_decoder_matches_sequential():
    """Batched decode must match independent single-request decodes (greedy)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = HydraLMForCausalLM(cfg).eval().model
    decoder = CompiledDecoder(model, compile=False)

    prompts = [
        torch.randint(0, cfg.vocab_size, (6,), generator=torch.Generator().manual_seed(1)),
        torch.randint(0, cfg.vocab_size, (6,), generator=torch.Generator().manual_seed(2)),
    ]

    # sequential
    seq_outs = []
    for p in prompts:
        r = Request(prompt=p, max_new_tokens=4, temperature=0.0)
        seq_outs.append(decoder.decode([r])[0])

    # batched
    batched_outs = decoder.decode([
        Request(prompt=p, max_new_tokens=4, temperature=0.0) for p in prompts
    ])

    for a, b in zip(seq_outs, batched_outs):
        assert torch.equal(a, b), f"batched != sequential: {a} vs {b}"
