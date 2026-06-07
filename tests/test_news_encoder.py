"""Tests for Fastformer, attention pooler, and news encoder (Module 1)."""

import torch

from newsrec.models.attention_pooler import AdditiveAttentionPooling
from newsrec.models.fastformer import FastformerConfig, FastformerEncoder
from newsrec.models.news_encoder import NewsEncoder


def test_fastformer_returns_sequence():
    enc = FastformerEncoder(FastformerConfig(hidden_size=32, num_hidden_layers=2,
                                             num_attention_heads=4)).eval()
    B, S, D = 4, 10, 32
    x = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    out = enc(x, mask)
    assert out.shape == (B, S, D)


def test_attention_pooler_shapes_and_mask():
    pool = AdditiveAttentionPooling(16)
    B, S, D = 3, 5, 16
    x = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    mask[:, 3:] = 0  # last 2 positions padded
    pooled, alpha = pool(x, mask)
    assert pooled.shape == (B, D)
    assert alpha.shape == (B, S)
    # Masked positions get ~0 attention weight.
    assert torch.allclose(alpha[:, 3:], torch.zeros(B, 2), atol=1e-6)
    # Weights over valid positions sum to 1.
    assert torch.allclose(alpha.sum(dim=1), torch.ones(B), atol=1e-5)


def test_pooler_ignores_padded_content():
    pool = AdditiveAttentionPooling(8).eval()
    B, S, D = 2, 6, 8
    base = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    mask[:, 4:] = 0
    pooled_a, _ = pool(base, mask)
    # Change the padded positions arbitrarily; pooled output must not change.
    perturbed = base.clone()
    perturbed[:, 4:] = torch.randn(B, 2, D)
    pooled_b, _ = pool(perturbed, mask)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-6)


def test_news_encoder_output_dim():
    enc = NewsEncoder(input_dim=64, hidden_size=32, num_layers=2, num_heads=4).eval()
    B, L = 5, 12
    word = torch.randn(B, L, 64)
    mask = torch.ones(B, L)
    h = enc(word, mask)
    assert h.shape == (B, 32)
    assert enc.output_dim == 32


def test_news_encoder_padding_invariance():
    torch.manual_seed(0)
    enc = NewsEncoder(input_dim=32, hidden_size=32, num_layers=2, num_heads=4).eval()
    B, L_valid = 2, 6
    valid = torch.randn(B, L_valid, 32)

    # Variant A: no padding.
    mask_a = torch.ones(B, L_valid)
    out_a = enc(valid, mask_a)

    # Variant B: same valid tokens + 4 padded positions at the end.
    pad = torch.randn(B, 4, 32)
    x_b = torch.cat([valid, pad], dim=1)
    mask_b = torch.cat([torch.ones(B, L_valid), torch.zeros(B, 4)], dim=1)
    out_b = enc(x_b, mask_b)

    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_news_encoder_no_projection_when_dims_match():
    enc = NewsEncoder(input_dim=48, hidden_size=48)
    assert enc.proj.__class__.__name__ == "Identity"
