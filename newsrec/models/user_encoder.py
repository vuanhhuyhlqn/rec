"""
user_encoder.py — Module 2 (user encoder)
==========================================

A bidirectional Fastformer over the user's history of news vectors.

It exposes BOTH:

* the per-position contextual hidden states ``[B, S, D]`` — consumed by the
  masked pre-training tasks (MIP / MAP), and
* a single pooled user vector ``z_u [B, D]`` produced by an additive attention
  pooler — used for recommendation scoring, the SP context/segment
  representations and the BSM sub-sequence embeddings.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from newsrec.models.attention_pooler import AdditiveAttentionPooling
from newsrec.models.fastformer import FastformerConfig, FastformerEncoder


class UserEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fastformer = FastformerEncoder(
            FastformerConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                hidden_dropout_prob=dropout,
                max_position_embeddings=max_position_embeddings,
            )
        )
        self.pooler = AdditiveAttentionPooling(hidden_size)

    def encode_sequence(
        self, news_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``[B, S, D]`` news vectors → ``[B, S, D]`` contextual states."""
        return self.fastformer(news_vectors, mask)

    def pool(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pooled, _ = self.pooler(sequence, mask)
        return pooled

    def forward(
        self, news_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(sequence_states [B, S, D], user_vector z_u [B, D])``."""
        sequence = self.encode_sequence(news_vectors, mask)
        z_u = self.pool(sequence, mask)
        return sequence, z_u

    @property
    def output_dim(self) -> int:
        return self.hidden_size
