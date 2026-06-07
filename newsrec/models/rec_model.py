"""
rec_model.py — Two-tower news recommender
==========================================

Wires Module 0 (PLM word embedder) → Module 1 (news encoder) → Module 2 (user
encoder + attention pooler) into an end-to-end two-tower model.

* ``encode_news``  : token ids ``[..., L]`` → news vectors ``[..., D]``
* ``encode_user``  : history token ids ``[B, S, L]`` → user vector ``z_u [B, D]``
  (also returns the per-position history states for pre-training).
* ``score``        : cosine similarity between ``z_u`` and candidate vectors.

The news vectors ``h_i`` and user vectors ``z_u`` produced here are exactly
the objects consumed by the pre-training tasks and PAAC losses in later tasks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from newsrec.models.news_encoder import NewsEncoder
from newsrec.models.plm_encoder import PLMEncoder
from newsrec.models.user_encoder import UserEncoder


DEFAULT_MODEL_CONFIG = {
    "plm": {
        "model_name": "bert-base-uncased",
        "pretrained": True,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
    "model_dim": 256,
    "news_encoder": {"num_layers": 2, "num_heads": 8, "dropout": 0.1},
    "user_encoder": {"num_layers": 2, "num_heads": 8, "dropout": 0.1},
    "score": {"type": "cosine", "temperature": 1.0},
    "max_title_len": 64,
    "max_history_len": 50,
}


class NewsRecModel(nn.Module):
    def __init__(
        self,
        plm: PLMEncoder,
        news_encoder: NewsEncoder,
        user_encoder: UserEncoder,
        score_type: str = "cosine",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.plm = plm
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.score_type = score_type
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Encoders                                                           #
    # ------------------------------------------------------------------ #
    def encode_news(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """``[..., L]`` token ids → ``[..., D]`` news vectors (any leading dims).

        Padding rows (no real tokens, e.g. empty history slots when the user has
        fewer than ``max_history`` clicks) are skipped entirely instead of being
        pushed through BERT — those rows are masked out downstream anyway, and
        encoding them dominates the cost when histories are short. Their output
        is left as zeros.
        """
        lead_shape = input_ids.shape[:-1]
        seq_len = input_ids.shape[-1]
        flat_ids = input_ids.reshape(-1, seq_len)
        flat_mask = attention_mask.reshape(-1, seq_len)
        n_rows = flat_ids.shape[0]
        out_dim = self.news_encoder.output_dim

        valid = flat_mask.sum(dim=-1) > 0  # rows that contain at least one real token
        if bool(valid.all()):
            word_emb, word_mask = self.plm(flat_ids, flat_mask)
            news_vec = self.news_encoder(word_emb, word_mask)  # [N, D]
        elif bool(valid.any()):
            word_emb, word_mask = self.plm(flat_ids[valid], flat_mask[valid])
            sub = self.news_encoder(word_emb, word_mask)       # [N_valid, D]
            # Match the encoder output dtype (e.g. bf16 under autocast).
            news_vec = sub.new_zeros((n_rows, out_dim))
            news_vec[valid] = sub
        else:
            news_vec = flat_ids.new_zeros((n_rows, out_dim), dtype=torch.float)
        return news_vec.reshape(*lead_shape, out_dim)

    def encode_user(
        self,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ``[B, S, L]`` history token ids → ``(history_states [B, S, D], z_u [B, D])``.

        ``history_mask`` (``[B, S]``, 1 = real click, 0 = padding) marks valid
        history positions; if omitted, all positions are treated as valid.
        """
        history_vecs = self.encode_news(history_input_ids, history_attention_mask)
        if history_mask is None:
            history_mask = torch.ones(
                history_vecs.shape[:2], device=history_vecs.device
            )
        sequence, z_u = self.user_encoder(history_vecs, history_mask)
        return sequence, z_u

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #
    def score(self, z_u: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        """
        Score user against candidates.

        ``z_u``                : ``[B, D]``
        ``candidate_vectors``  : ``[B, K, D]``  (or ``[B, D]`` for a single cand)
        Returns                : ``[B, K]``     (or ``[B]``)
        """
        single = candidate_vectors.dim() == 2
        if single:
            candidate_vectors = candidate_vectors.unsqueeze(1)

        if self.score_type == "cosine":
            u = F.normalize(z_u, dim=-1).unsqueeze(1)          # [B, 1, D]
            c = F.normalize(candidate_vectors, dim=-1)         # [B, K, D]
            scores = (u * c).sum(-1)                            # [B, K]
        elif self.score_type == "dot":
            scores = torch.bmm(candidate_vectors, z_u.unsqueeze(-1)).squeeze(-1)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown score_type '{self.score_type}'")

        scores = scores / self.temperature
        return scores.squeeze(1) if single else scores

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Full forward producing candidate scores ``[B, K]``.

        Expected keys:
            history_input_ids       [B, S, L]
            history_attention_mask  [B, S, L]
            history_mask            [B, S]            (optional)
            candidate_input_ids     [B, K, L]
            candidate_attention_mask[B, K, L]
        """
        _, z_u = self.encode_user(
            batch["history_input_ids"],
            batch["history_attention_mask"],
            batch.get("history_mask"),
        )
        cand_vecs = self.encode_news(
            batch["candidate_input_ids"], batch["candidate_attention_mask"]
        )
        return self.score(z_u, cand_vecs)

    @property
    def model_dim(self) -> int:
        return self.user_encoder.hidden_size


# --------------------------------------------------------------------------- #
# Builder                                                                     #
# --------------------------------------------------------------------------- #
def build_rec_model(config: Optional[dict] = None) -> NewsRecModel:
    """Construct a :class:`NewsRecModel` from a (partial) config dict."""
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if config:
        # shallow-merge top level, deep-merge known nested dicts
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                merged = dict(cfg[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value

    plm = PLMEncoder(**cfg["plm"])
    model_dim = cfg["model_dim"]
    news_encoder = NewsEncoder(
        input_dim=plm.output_dim,
        hidden_size=model_dim,
        num_layers=cfg["news_encoder"]["num_layers"],
        num_heads=cfg["news_encoder"]["num_heads"],
        dropout=cfg["news_encoder"]["dropout"],
        max_position_embeddings=cfg.get("max_title_len", 512),
    )
    user_encoder = UserEncoder(
        hidden_size=model_dim,
        num_layers=cfg["user_encoder"]["num_layers"],
        num_heads=cfg["user_encoder"]["num_heads"],
        dropout=cfg["user_encoder"]["dropout"],
        max_position_embeddings=cfg.get("max_history_len", 512),
    )
    return NewsRecModel(
        plm=plm,
        news_encoder=news_encoder,
        user_encoder=user_encoder,
        score_type=cfg["score"]["type"],
        temperature=cfg["score"].get("temperature", 1.0),
    )
