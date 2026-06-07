"""
infonce.py
==========

InfoNCE / contrastive loss helpers shared by the pre-training tasks.

* :func:`info_nce_inbatch` — symmetric in-batch InfoNCE: positives lie on the
  diagonal of the ``query · key`` similarity matrix, all other columns are
  negatives.  Used by MIP, SP, BSM.
* :func:`info_nce_against_table` — classify each anchor against a fixed table
  of candidate embeddings (e.g. the category table); the true index is the
  positive.  Used by AAP and MAP.

Embeddings are L2-normalised before the dot product so the temperature ``tau``
behaves consistently (cosine-similarity InfoNCE).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_inbatch(query: torch.Tensor, key: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    In-batch InfoNCE.

    ``query`` / ``key`` are ``[N, D]``; row ``i`` of ``query`` is matched
    against row ``i`` of ``key`` (positive) versus all other rows (negatives).
    Returns the mean of the two symmetric cross-entropies.
    """
    n = query.shape[0]
    if n < 2:
        return torch.zeros((), device=query.device)
    q = F.normalize(query, dim=-1)
    k = F.normalize(key, dim=-1)
    logits = (q @ k.t()) / tau            # [N, N]
    labels = torch.arange(n, device=query.device)
    loss_qk = F.cross_entropy(logits, labels)
    loss_kq = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_qk + loss_kq)


def info_nce_against_table(
    anchor: torch.Tensor,
    labels: torch.Tensor,
    table: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Classify ``anchor [N, D]`` against ``table [C, D]`` candidate embeddings.

    ``labels [N]`` give the index of the positive candidate for each anchor;
    every other table row is a negative.  Equivalent to InfoNCE where the
    negatives are the other categories.
    """
    if anchor.shape[0] == 0:
        return torch.zeros((), device=anchor.device)
    a = F.normalize(anchor, dim=-1)
    t = F.normalize(table, dim=-1)
    logits = (a @ t.t()) / tau            # [N, C]
    return F.cross_entropy(logits, labels)
