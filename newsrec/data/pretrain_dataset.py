"""
pretrain_dataset.py
===================

S3Rec-style self-supervised pre-training samples built from per-user click
sequences.

Unlike S3Rec (which embeds item *ids*), our items are *encoded* by the PLM +
news encoder, so this dataset emits the **news token tensors** for each history
position plus a set of boolean masks / targets that the pre-training losses
(Task 11) consume:

Per sample (fixed sequence length ``S``):

* ``input_ids``      ``[S, L]`` / ``attention_mask`` ``[S, L]`` — news tokens.
* ``seq_mask``       ``[S]``    — 1 = real history item, 0 = padding.
* ``mip_mask``       ``[S]``    — 1 = position masked for MIP / MAP.
* ``category``       ``[S]``    — category index per position (AAP / MAP target).
* ``segment_mask``   ``[S]``    — 1 = position inside the SP masked segment.
* ``context_mask``   ``[S]``    — ``seq_mask AND NOT segment_mask`` (SP context).
* ``bsm_a_mask`` / ``bsm_b_mask`` ``[S]`` — the two non-overlapping temporal
  sub-sequences for BSM.

The learnable ``[MASK]`` embedding itself lives at the news-vector level and is
applied inside the pre-training model/losses using ``mip_mask`` — the dataset
only flags *which* positions are masked.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def build_user_sequences(
    impressions: Sequence,
    min_len: int = 3,
    in_table=None,
) -> List[Tuple[str, List[str]]]:
    """
    Build one click sequence per user.

    For each user we keep their *longest* observed history (the most recent
    impression's history is the longest in MIND).  Items absent from the token
    table (``in_table`` callable) are dropped.  Sequences shorter than
    ``min_len`` are discarded.
    """
    best: Dict[str, List[str]] = {}
    for imp in impressions:
        hist = imp.history
        if in_table is not None:
            hist = [n for n in hist if in_table(n)]
        if len(hist) > len(best.get(imp.user_id, [])):
            best[imp.user_id] = hist
    return [(u, h) for u, h in best.items() if len(h) >= min_len]


class PretrainDataset(Dataset):
    def __init__(
        self,
        user_sequences: Sequence[Tuple[str, List[str]]],
        news_tokens: Mapping,
        category_ids: Mapping[str, int],
        max_seq_len: int = 50,
        mask_prob: float = 0.15,
        seed: int = 42,
    ):
        self.sequences = [seq for _, seq in user_sequences]
        self.news_tokens = news_tokens
        self.category_ids = category_ids
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.max_len = news_tokens.max_len
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    # ------------------------------------------------------------------ #
    def _cat(self, nid: str) -> int:
        return int(self.category_ids.get(nid, 0))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[index][-self.max_seq_len:]
        n = len(seq)
        S = self.max_seq_len
        L = self.max_len

        input_ids = torch.zeros(S, L, dtype=torch.long)
        attention_mask = torch.zeros(S, L, dtype=torch.long)
        seq_mask = torch.zeros(S, dtype=torch.float)
        mip_mask = torch.zeros(S, dtype=torch.float)
        category = torch.zeros(S, dtype=torch.long)

        for i, nid in enumerate(seq):
            tok = self.news_tokens.get(nid)
            input_ids[i] = torch.tensor(tok["input_ids"], dtype=torch.long)
            attention_mask[i] = torch.tensor(tok["attention_mask"], dtype=torch.long)
            seq_mask[i] = 1.0
            category[i] = self._cat(nid)

        # --- MIP / MAP masking -------------------------------------------------
        masked_positions = [i for i in range(n) if self._rng.random() < self.mask_prob]
        if not masked_positions and n > 0:
            masked_positions = [n - 1]  # always mask at least one (S3Rec-style)
        for i in masked_positions:
            mip_mask[i] = 1.0

        # --- SP segment --------------------------------------------------------
        segment_mask = torch.zeros(S, dtype=torch.float)
        if n >= 2:
            seg_len = self._rng.randint(1, max(1, n // 2))
            start = self._rng.randint(0, n - seg_len)
            segment_mask[start:start + seg_len] = 1.0
        elif n == 1:
            segment_mask[0] = 1.0
        context_mask = seq_mask * (1.0 - segment_mask)

        # --- BSM temporal split ------------------------------------------------
        bsm_a_mask = torch.zeros(S, dtype=torch.float)
        bsm_b_mask = torch.zeros(S, dtype=torch.float)
        if n >= 2:
            mid = n // 2
            bsm_a_mask[:mid] = 1.0
            bsm_b_mask[mid:n] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_mask": seq_mask,
            "mip_mask": mip_mask,
            "category": category,
            "segment_mask": segment_mask,
            "context_mask": context_mask,
            "bsm_a_mask": bsm_a_mask,
            "bsm_b_mask": bsm_b_mask,
        }
