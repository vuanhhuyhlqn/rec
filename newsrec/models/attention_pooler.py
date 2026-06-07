"""
attention_pooler.py
====================

Additive attention pooling: ``[B, S, D]`` sequence of vectors → ``[B, D]``
single vector, respecting a padding mask.

Used by the news encoder (pool word states) and the user encoder
(pool history states into the user vector ``z_u``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class AdditiveAttentionPooling(nn.Module):
    """Masked additive (Bahdanau-style) attention pooling."""

    def __init__(self, hidden_size: int, attention_dim: Optional[int] = None):
        super().__init__()
        attention_dim = attention_dim or hidden_size
        self.fc1 = nn.Linear(hidden_size, attention_dim)
        self.fc2 = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:    ``[B, S, D]`` sequence of vectors.
        mask: ``[B, S]`` with 1 = valid, 0 = padding (optional).

        Returns
        -------
        (pooled ``[B, D]``, attention weights ``[B, S]``)
        """
        scores = self.fc2(self.tanh(self.fc1(x))).squeeze(-1)  # [B, S]
        if mask is not None:
            mask = mask.to(dtype=scores.dtype)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # Guard against all-masked rows producing NaNs.
        alpha = torch.softmax(scores, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        pooled = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled, alpha
