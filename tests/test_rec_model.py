"""Tests for newsrec.models.rec_model (two-tower)."""

import torch

from newsrec.models.rec_model import build_rec_model


SMALL_PLM = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=64, max_position_embeddings=64, vocab_size=200)


def _build(model_dim=24):
    cfg = {
        "plm": {"pretrained": False, "use_lora": True, "lora_r": 4,
                "small_config": SMALL_PLM},
        "model_dim": model_dim,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.1},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.1},
        "max_title_len": 16,
        "max_history_len": 20,
    }
    return build_rec_model(cfg).eval()


def test_encode_news_arbitrary_leading_dims():
    model = _build(model_dim=24)
    B, K, L = 3, 5, 12
    ids = torch.randint(0, 200, (B, K, L))
    mask = torch.ones(B, K, L, dtype=torch.long)
    vecs = model.encode_news(ids, mask)
    assert vecs.shape == (B, K, 24)


def test_encode_user_shapes():
    model = _build(model_dim=24)
    B, S, L = 4, 6, 12
    hist_ids = torch.randint(0, 200, (B, S, L))
    hist_attn = torch.ones(B, S, L, dtype=torch.long)
    hist_mask = torch.ones(B, S)
    seq, z_u = model.encode_user(hist_ids, hist_attn, hist_mask)
    assert seq.shape == (B, S, 24)
    assert z_u.shape == (B, 24)


def test_forward_scores_shape():
    model = _build()
    B, S, K, L = 2, 5, 4, 12
    batch = {
        "history_input_ids": torch.randint(0, 200, (B, S, L)),
        "history_attention_mask": torch.ones(B, S, L, dtype=torch.long),
        "history_mask": torch.ones(B, S),
        "candidate_input_ids": torch.randint(0, 200, (B, K, L)),
        "candidate_attention_mask": torch.ones(B, K, L, dtype=torch.long),
    }
    scores = model(batch)
    assert scores.shape == (B, K)


def test_cosine_bounds_and_identity():
    model = _build(model_dim=16)
    B, K, D = 3, 4, 16
    z_u = torch.randn(B, D)
    cand = torch.randn(B, K, D)
    scores = model.score(z_u, cand)
    assert scores.shape == (B, K)
    assert torch.all(scores <= 1.0 + 1e-5) and torch.all(scores >= -1.0 - 1e-5)

    # Identical user / candidate vectors → cosine ≈ 1.
    same = torch.randn(B, D)
    s = model.score(same, same.unsqueeze(1))
    assert torch.allclose(s.squeeze(1), torch.ones(B), atol=1e-5)


def test_single_candidate_score_shape():
    model = _build(model_dim=16)
    z_u = torch.randn(2, 16)
    cand = torch.randn(2, 16)  # single candidate, no K dim
    s = model.score(z_u, cand)
    assert s.shape == (2,)
