"""
collate.py
==========

Batch collation helpers.  Both the fine-tune and pre-train datasets emit
fixed-shape tensors per sample, so collation is just a per-key ``torch.stack``.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def stack_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of ``{key: tensor}`` samples into ``{key: [B, ...]}``."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}
