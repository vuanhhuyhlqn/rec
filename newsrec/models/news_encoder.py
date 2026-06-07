"""
news_encoder.py — Module 1 (news encoder)
==========================================

Turns the word embeddings ``[B, L, D_in]`` produced by Module 0 into a single
news vector ``h_i [B, D_out]`` via a Fastformer + additive attention pooling
(+ optional projection to the shared model dimension).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from newsrec.models.attention_pooler import AdditiveAttentionPooling
from newsrec.models.fastformer import FastformerConfig, FastformerEncoder


class NewsEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.fastformer = FastformerEncoder(
            FastformerConfig(
                hidden_size=input_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                hidden_dropout_prob=dropout,
                max_position_embeddings=max_position_embeddings,
            )
        )
        self.pooler = AdditiveAttentionPooling(input_dim)
        self.proj = (
            nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size)
        )

    def forward(
        self, word_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``[B, L, D_in]`` (+ ``[B, L]`` mask) → news vector ``[B, D_out]``."""
        seq = self.fastformer(word_embeddings, mask)  # [B, L, D_in]
        pooled, _ = self.pooler(seq, mask)            # [B, D_in]
        return self.proj(pooled)                      # [B, D_out]

    @property
    def output_dim(self) -> int:
        return self.hidden_size
