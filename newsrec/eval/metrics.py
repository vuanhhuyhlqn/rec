"""
metrics.py
==========

Per-impression ranking metrics for MIND: AUC, MRR, nDCG@5, nDCG@10.

Each metric is computed *within* a single impression (one user, one candidate
list with binary click labels) and then averaged across impressions — this is
the standard MIND evaluation protocol.  Implementations are cross-checked
against the formulas used in the Legommenders ``utils/metrics.py`` (which in
turn follow the official MIND scoring script).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _as_arrays(scores: Sequence[float], labels: Sequence[int]):
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC for a single impression (rank-based, handles ties)."""
    s, y = _as_arrays(scores, labels)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined; excluded from the average
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties to keep AUC unbiased.
    _assign_tie_ranks(s, ranks)
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def mrr_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mean Reciprocal Rank (MIND variant: averaged over all positives)."""
    s, y = _as_arrays(scores, labels)
    order = np.argsort(s)[::-1]
    y_ranked = y[order]
    rr = y_ranked / (np.arange(len(y_ranked)) + 1)
    denom = y.sum()
    if denom == 0:
        return float("nan")
    return float(rr.sum() / denom)


def _dcg(y_ranked: np.ndarray, k: int) -> float:
    y_k = y_ranked[:k]
    gains = (2 ** y_k - 1) / np.log2(np.arange(2, len(y_k) + 2))
    return float(gains.sum())


def ndcg_score(scores: Sequence[float], labels: Sequence[int], k: int = 10) -> float:
    """nDCG@k for a single impression."""
    s, y = _as_arrays(scores, labels)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(s)[::-1]
    dcg = _dcg(y[order], k)
    ideal = _dcg(np.sort(y)[::-1], k)
    if ideal == 0:
        return float("nan")
    return dcg / ideal


METRIC_FNS = {
    "auc": lambda s, y: auc_score(s, y),
    "mrr": lambda s, y: mrr_score(s, y),
    "ndcg@5": lambda s, y: ndcg_score(s, y, 5),
    "ndcg@10": lambda s, y: ndcg_score(s, y, 10),
}


def compute_impression_metrics(
    scores_list: List[Sequence[float]],
    labels_list: List[Sequence[int]],
    metrics: Sequence[str] = ("auc", "mrr", "ndcg@5", "ndcg@10"),
) -> Dict[str, float]:
    """
    Average each metric over impressions, skipping impressions where the
    metric is undefined (NaN, e.g. all-positive or all-negative for AUC).
    """
    accum: Dict[str, List[float]] = {m: [] for m in metrics}
    for scores, labels in zip(scores_list, labels_list):
        for m in metrics:
            value = METRIC_FNS[m](scores, labels)
            if not np.isnan(value):
                accum[m].append(value)
    return {m: float(np.mean(v)) if v else float("nan") for m, v in accum.items()}
