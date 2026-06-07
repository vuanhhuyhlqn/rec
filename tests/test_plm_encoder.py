"""Tests for newsrec.models.plm_encoder (Module 0)."""

import torch

from newsrec.models.plm_encoder import PLMEncoder


SMALL = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
             intermediate_size=128, max_position_embeddings=64, vocab_size=200)


def _make_encoder(use_lora=True):
    return PLMEncoder(pretrained=False, use_lora=use_lora, lora_r=4,
                      small_config=SMALL)


def test_output_shape():
    enc = _make_encoder()
    B, L = 3, 16
    ids = torch.randint(0, 200, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    emb, out_mask = enc(ids, mask)
    assert emb.shape == (B, L, 64)
    assert out_mask.shape == (B, L)
    assert enc.output_dim == 64


def test_lora_only_trainable_by_default():
    enc = _make_encoder(use_lora=True)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    # All trainable params must be LoRA params when frozen-backbone.
    assert trainable, "expected some trainable LoRA params"
    assert all("lora_" in n for n in trainable)
    assert enc.frozen_layers == enc.num_layers


def test_gradual_unfreeze_opens_top_layers():
    enc = _make_encoder(use_lora=True)
    enc.set_trainable_layers(2)  # open top 2 of 4 layers
    assert enc.frozen_layers == enc.num_layers - 2

    open_base = [
        n for n, p in enc.named_parameters()
        if p.requires_grad and "lora_" not in n
    ]
    # Opened base params should belong to layers 2 and 3 only (top 2).
    assert open_base, "expected base params to be unfrozen"
    for name in open_base:
        idx = enc._layer_index_of(name)
        assert idx is not None and idx >= 2


def test_full_unfreeze_no_lora():
    enc = _make_encoder(use_lora=False)
    enc.set_trainable_layers(enc.num_layers)
    # With LoRA disabled and full unfreeze, embeddings should also be trainable.
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert any("embeddings" in n for n in names)
    assert enc.frozen_layers == 0


def test_trainable_param_count_monotonic():
    enc = _make_encoder(use_lora=True)
    base = enc.num_trainable_parameters()
    enc.set_trainable_layers(1)
    one = enc.num_trainable_parameters()
    enc.set_trainable_layers(4)
    four = enc.num_trainable_parameters()
    assert base < one < four


def test_gradient_flows_to_lora():
    enc = _make_encoder(use_lora=True)
    ids = torch.randint(0, 200, (2, 8))
    emb, _ = enc(ids)
    emb.sum().backward()
    grads = [p.grad for n, p in enc.named_parameters()
             if p.requires_grad and "lora_B" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
