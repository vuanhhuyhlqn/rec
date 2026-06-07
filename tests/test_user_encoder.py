"""Tests for newsrec.models.user_encoder (Module 2)."""

import torch

from newsrec.models.user_encoder import UserEncoder


def test_user_encoder_dual_output_shapes():
    enc = UserEncoder(hidden_size=32, num_layers=2, num_heads=4).eval()
    B, S, D = 4, 8, 32
    news = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    seq, z_u = enc(news, mask)
    assert seq.shape == (B, S, D)
    assert z_u.shape == (B, D)
    assert enc.output_dim == 32


def test_user_encoder_masked_pooling_invariance():
    torch.manual_seed(0)
    enc = UserEncoder(hidden_size=32, num_layers=2, num_heads=4).eval()
    B, S_valid = 2, 5
    valid = torch.randn(B, S_valid, 32)
    out_a = enc.pool(enc.encode_sequence(valid, torch.ones(B, S_valid)),
                     torch.ones(B, S_valid))

    pad = torch.randn(B, 3, 32)
    x_b = torch.cat([valid, pad], dim=1)
    mask_b = torch.cat([torch.ones(B, S_valid), torch.zeros(B, 3)], dim=1)
    seq_b = enc.encode_sequence(x_b, mask_b)
    out_b = enc.pool(seq_b, mask_b)
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_user_encoder_gradient_flow():
    enc = UserEncoder(hidden_size=16, num_layers=1, num_heads=4)
    news = torch.randn(3, 6, 16, requires_grad=True)
    _, z_u = enc(news, torch.ones(3, 6))
    z_u.sum().backward()
    assert news.grad is not None and news.grad.abs().sum() > 0
    # encoder params receive gradient too
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_per_position_states_accessible():
    enc = UserEncoder(hidden_size=24, num_layers=2, num_heads=4).eval()
    B, S = 2, 7
    seq = enc.encode_sequence(torch.randn(B, S, 24), torch.ones(B, S))
    # Per-position states needed by MIP/MAP.
    masked_pos = torch.tensor([2, 5])
    states_at_pos = seq[torch.arange(B), masked_pos]
    assert states_at_pos.shape == (B, 24)
