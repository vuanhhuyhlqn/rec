"""
pretrain_model.py
=================

:class:`PretrainModule` wraps a :class:`~newsrec.models.rec_model.NewsRecModel`
backbone (Modules 0/1/2) and adds the parameters the self-supervised tasks
need:

* a learnable category embedding table (AAP / MAP), and
* a learnable ``[MASK]`` news-vector token (MIP / MAP).

It exposes :meth:`compute_losses` which, given a batch from
:class:`~newsrec.data.pretrain_dataset.PretrainDataset`, runs the shared
encoders once and returns the enabled task losses plus the weighted ``total``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn

from newsrec.losses.pretrain_losses import (
    aap_loss,
    bsm_loss,
    map_loss,
    mip_loss,
    sp_loss,
    task_weights,
)


class PretrainModule(nn.Module):
    def __init__(
        self,
        model,
        num_categories: int,
        enabled_tasks: Iterable[str] = ("aap", "mip", "map", "sp", "bsm"),
        weights: Optional[dict] = None,
        tau: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.enabled = list(enabled_tasks)
        self.weights = weights or {t: 1.0 for t in self.enabled}
        self.tau = tau

        dim = model.model_dim
        self.category_embeddings = nn.Embedding(num_categories, dim)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)

    # ------------------------------------------------------------------ #
    def _encode_history(self, batch) -> torch.Tensor:
        return self.model.encode_news(batch["input_ids"], batch["attention_mask"])

    def _seq(self, vecs, mask):
        return self.model.user_encoder.encode_sequence(vecs, mask)

    def _pool(self, states, mask):
        return self.model.user_encoder.pool(states, mask)

    # ------------------------------------------------------------------ #
    def compute_losses(self, batch) -> Dict[str, torch.Tensor]:
        device = self.mask_token.device
        batch = {k: v.to(device) for k, v in batch.items()}

        h_seq = self._encode_history(batch)         # [B, S, D]
        seq_mask = batch["seq_mask"]                # [B, S]
        category = batch["category"]                # [B, S]
        cat_table = self.category_embeddings.weight  # [C, D]

        losses: Dict[str, torch.Tensor] = {}

        # ---- AAP (item level) -------------------------------------------- #
        if "aap" in self.enabled:
            valid = seq_mask > 0
            losses["aap"] = aap_loss(h_seq[valid], category[valid], cat_table, self.tau)

        # ---- MIP / MAP (need masked contextual states) ------------------- #
        if "mip" in self.enabled or "map" in self.enabled:
            mip_mask = batch["mip_mask"] > 0
            masked_input = h_seq.clone()
            masked_input[mip_mask] = self.mask_token.to(masked_input.dtype)
            ctx = self._seq(masked_input, seq_mask)  # [B, S, D]
            ctx_masked = ctx[mip_mask]               # [M, D]
            if "mip" in self.enabled:
                losses["mip"] = mip_loss(ctx_masked, h_seq[mip_mask], self.tau)
            if "map" in self.enabled:
                losses["map"] = map_loss(ctx_masked, category[mip_mask], cat_table, self.tau)

        # ---- SP (segment vs context) ------------------------------------- #
        if "sp" in self.enabled:
            seg_mask = batch["segment_mask"]
            ctx_mask = batch["context_mask"]
            keep = (seg_mask.sum(1) > 0) & (ctx_mask.sum(1) > 0)
            if keep.any():
                # Context: replace segment positions with the [MASK] token.
                masked_seg_input = h_seq.clone()
                masked_seg_input[seg_mask > 0] = self.mask_token.to(h_seq.dtype)
                ctx_states = self._seq(masked_seg_input, seq_mask)
                context_repr = self._pool(ctx_states, ctx_mask)[keep]
                # Segment: encode the segment positions independently.
                seg_states = self._seq(h_seq, seg_mask)
                segment_repr = self._pool(seg_states, seg_mask)[keep]
                losses["sp"] = sp_loss(context_repr, segment_repr, self.tau)
            else:
                losses["sp"] = torch.zeros((), device=device)

        # ---- BSM (two temporal sub-sequences) ---------------------------- #
        if "bsm" in self.enabled:
            a_mask = batch["bsm_a_mask"]
            b_mask = batch["bsm_b_mask"]
            keep = (a_mask.sum(1) > 0) & (b_mask.sum(1) > 0)
            if keep.any():
                z_a = self._pool(self._seq(h_seq, a_mask), a_mask)[keep]
                z_b = self._pool(self._seq(h_seq, b_mask), b_mask)[keep]
                losses["bsm"] = bsm_loss(z_a, z_b, self.tau)
            else:
                losses["bsm"] = torch.zeros((), device=device)

        # ---- weighted total ---------------------------------------------- #
        total = torch.zeros((), device=device)
        for name, value in losses.items():
            total = total + self.weights.get(name, 1.0) * value
        losses["total"] = total
        return losses
