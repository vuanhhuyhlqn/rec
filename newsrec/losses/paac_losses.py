"""
paac_losses.py
==============

Loss components for the PAAC fine-tuning stage.

This task (Task 8) implements the main recommendation loss:

* :func:`bpr_loss` — Bayesian Personalised Ranking over (user, pos, neg)
  triplets.

The popularity-aware alignment (``L_sa``) and re-weighting contrastive
(``L_cl``) losses are added in Task 9.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    BPR loss ``-mean(log(sigmoid(s_pos - s_neg)))``.

    ``pos_scores`` / ``neg_scores`` are ``[B]`` (one positive and one negative
    score per triplet).  Uses ``logsigmoid`` for numerical stability.
    """
    diff = pos_scores - neg_scores
    return -F.logsigmoid(diff).mean()


def l2_regularization(parameters) -> torch.Tensor:
    """Sum of squared L2 norms of the given parameters (the ``||Theta||^2`` term)."""
    total = None
    for p in parameters:
        if p.requires_grad:
            term = p.pow(2).sum()
            total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total


# --------------------------------------------------------------------------- #
# L_sa — Popularity-Aware Supervised Alignment                                #
# --------------------------------------------------------------------------- #
def supervised_alignment_loss(
    history_vecs: torch.Tensor,
    history_mask: torch.Tensor,
    history_pop: torch.Tensor,
    ratio: float = 0.5,
) -> torch.Tensor:
    """
    PAAC ``L_sa``.

    For each user, split the (valid) history items into a popular group and an
    unpopular group by global popularity, then accumulate the mean pairwise L2
    distance between popular and unpopular *news embeddings* (Module-1 output),
    normalised by the number of history items ``1/|I_u|``.

    Parameters
    ----------
    history_vecs : ``[B, S, D]`` news embeddings of the history items.
    history_mask : ``[B, S]`` 1 = valid history position, 0 = padding.
    history_pop  : ``[B, S]`` global popularity (count) per history item.
    ratio        : fraction assigned to the popular group.

    Returns a scalar (averaged over users in the batch for scale stability).
    """
    B = history_vecs.shape[0]
    device = history_vecs.device
    total = torch.zeros((), device=device)
    counted = 0

    for b in range(B):
        valid = history_mask[b] > 0
        n = int(valid.sum().item())
        if n < 2:
            continue
        vecs = history_vecs[b][valid]          # [n, D]
        pops = history_pop[b][valid]           # [n]
        order = torch.argsort(pops, descending=True)
        k = max(1, min(n - 1, int(round(n * ratio))))
        pop_idx = order[:k]
        unpop_idx = order[k:]
        if pop_idx.numel() == 0 or unpop_idx.numel() == 0:
            continue
        pop_vecs = vecs[pop_idx]               # [P, D]
        unpop_vecs = vecs[unpop_idx]           # [U, D]
        dists = torch.cdist(pop_vecs, unpop_vecs, p=2)  # [P, U]
        total = total + dists.sum() / n
        counted += 1

    if counted == 0:
        return total
    return total / counted


# --------------------------------------------------------------------------- #
# L_cl — Re-weighting Contrastive Learning                                    #
# --------------------------------------------------------------------------- #
def augment_views(
    h: torch.Tensor,
    dropout_p: float = 0.1,
    noise_std: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two augmented views of ``h`` via feature dropout + Gaussian noise."""

    def _aug(x: torch.Tensor) -> torch.Tensor:
        out = x
        if dropout_p > 0:
            mask = (torch.rand_like(out) > dropout_p).to(out.dtype)
            out = out * mask / (1.0 - dropout_p)
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out

    return _aug(h), _aug(h)


def _group_indices_by_pop(pop_values: torch.Tensor, x_percent: float):
    """Top-x% popularity → (popular_idx, unpopular_idx) as LongTensors."""
    n = pop_values.shape[0]
    k = int(round(n * x_percent / 100.0))
    k = max(0, min(n, k))
    order = torch.argsort(pop_values, descending=True)
    return order[:k], order[k:]


def _anchor_group_loss(
    sim: torch.Tensor,
    anchor_idx: torch.Tensor,
    primary_idx: torch.Tensor,
    secondary_idx: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    InfoNCE for anchors in ``anchor_idx`` where positives are the matching
    view (diagonal), the primary group contributes full weight to the
    denominator and the secondary group is weighted by ``beta``.
    """
    if anchor_idx.numel() == 0:
        return torch.zeros((), device=sim.device)

    losses = []
    log_beta = torch.log(torch.tensor(beta, device=sim.device)) if beta > 0 else None
    for i in anchor_idx.tolist():
        pos = sim[i, i]
        denom_terms = [sim[i, primary_idx]]
        if log_beta is not None and secondary_idx.numel() > 0:
            denom_terms.append(sim[i, secondary_idx] + log_beta)
        denom = torch.logsumexp(torch.cat(denom_terms), dim=0)
        losses.append(-(pos - denom))
    return torch.stack(losses).mean()


def reweighting_contrastive_loss(
    item_vecs: torch.Tensor,
    pop_values: torch.Tensor,
    x_percent: float = 50.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    tau: float = 0.1,
    dropout_p: float = 0.1,
    noise_std: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    PAAC ``L_cl`` (item-level).

    Two augmented views of the batch item embeddings are created; the batch is
    split into popular / unpopular groups by global popularity (top-x%).  A
    ``beta``-reweighted InfoNCE is computed with popular items as anchors and
    with unpopular items as anchors, combined by ``gamma``.
    """
    n = item_vecs.shape[0]
    if n < 2:
        return torch.zeros((), device=item_vecs.device)

    h1, h2 = augment_views(item_vecs, dropout_p, noise_std, generator)
    h1 = torch.nn.functional.normalize(h1, dim=-1)
    h2 = torch.nn.functional.normalize(h2, dim=-1)
    sim = (h1 @ h2.t()) / tau  # [N, N]

    pop_idx, unpop_idx = _group_indices_by_pop(pop_values, x_percent)

    l_pop = _anchor_group_loss(sim, pop_idx, pop_idx, unpop_idx, beta)
    l_unpop = _anchor_group_loss(sim, unpop_idx, unpop_idx, pop_idx, beta)
    return gamma * l_pop + (1.0 - gamma) * l_unpop
