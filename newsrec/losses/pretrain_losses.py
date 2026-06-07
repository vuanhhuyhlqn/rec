"""
pretrain_losses.py
==================

The five S3Rec-style self-supervised pre-training losses, all formulated as
InfoNCE (per the project spec):

* AAP — Associated Attribute Prediction  (item level)
* MIP — Masked Item Prediction           (sequence level)
* MAP — Masked Attribute Prediction      (sequence level)
* SP  — Segment Prediction               (sequence level)
* BSM — Behavior Sequence Matching       (user level)

Each function operates on already-encoded tensors (news vectors / contextual
states / category table) so they are easy to unit-test.  Orchestration of the
encoders + the learnable ``[MASK]`` token lives in
:class:`newsrec.models.pretrain_model.PretrainModule`.

``PRETRAIN_TASKS`` is the registry used for config-driven task selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, List

import torch

from newsrec.losses.infonce import info_nce_against_table, info_nce_inbatch


PRETRAIN_TASKS: List[str] = ["aap", "mip", "map", "sp", "bsm"]


def aap_loss(news_vecs, categories, category_table, tau: float = 0.1) -> torch.Tensor:
    """AAP: align each item's news vector with its true category embedding."""
    return info_nce_against_table(news_vecs, categories, category_table, tau)


def mip_loss(context_states, target_vecs, tau: float = 0.1) -> torch.Tensor:
    """MIP: match masked-position contextual state with the true news vector."""
    return info_nce_inbatch(context_states, target_vecs, tau)


def map_loss(context_states, categories, category_table, tau: float = 0.1) -> torch.Tensor:
    """MAP: predict the masked item's category from its contextual state."""
    return info_nce_against_table(context_states, categories, category_table, tau)


def sp_loss(context_repr, segment_repr, tau: float = 0.1) -> torch.Tensor:
    """SP: agreement between the surrounding context and the masked segment."""
    return info_nce_inbatch(context_repr, segment_repr, tau)


def bsm_loss(user_repr_a, user_repr_b, tau: float = 0.1) -> torch.Tensor:
    """BSM: match two temporal sub-sequences of the same user."""
    return info_nce_inbatch(user_repr_a, user_repr_b, tau)


def select_enabled_tasks(config) -> List[str]:
    """
    Resolve the enabled task list from a config.

    Accepts either a list ``["aap", "mip"]`` or a mapping
    ``{"aap": {"enabled": true, "weight": 1.0}, ...}``.  Unknown tasks raise.
    """
    if config is None:
        return []
    if isinstance(config, Mapping):
        enabled = []
        for name, spec in config.items():
            if isinstance(spec, Mapping):
                if spec.get("enabled", True):
                    enabled.append(name)
            elif spec:
                enabled.append(name)
        names = enabled
    else:
        names = list(config)

    for name in names:
        if name not in PRETRAIN_TASKS:
            raise ValueError(f"Unknown pre-train task '{name}'. Known: {PRETRAIN_TASKS}")
    return names


def task_weights(config, enabled: Iterable[str]) -> dict:
    """Per-task weights (default 1.0) from a mapping config."""
    weights = {}
    for name in enabled:
        w = 1.0
        if isinstance(config, Mapping):
            spec = config.get(name)
            if isinstance(spec, Mapping):
                w = float(spec.get("weight", 1.0))
        weights[name] = w
    return weights
