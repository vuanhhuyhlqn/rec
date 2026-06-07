"""
finetune_dataset.py
===================

BPR triplet dataset for PAAC fine-tuning.

For every impression with at least one clicked and one non-clicked candidate,
we emit triplets ``(history, i_pos, j_neg)`` where:

* ``i_pos`` is a clicked candidate of the impression, and
* ``j_neg`` is sampled from the *non-clicked* candidates of the **same**
  impression (in-impression negatives).

Each ``__getitem__`` returns fixed-shape token tensors so the default
collate stacks them directly into a batch.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class FinetuneTripletDataset(Dataset):
    def __init__(
        self,
        impressions: Sequence,
        news_tokens: Mapping,
        max_history: int = 50,
        negatives_per_pos: int = 1,
        seed: int = 42,
        popularity=None,
    ):
        self.news_tokens = news_tokens
        self.max_history = max_history
        self.max_len = news_tokens.max_len
        self.popularity = popularity
        self._rng = random.Random(seed)

        self.triplets: List[Tuple[List[str], str, str]] = []
        for imp in impressions:
            clicked = imp.clicked
            non_clicked = imp.non_clicked
            if not clicked or not non_clicked:
                continue
            history = [n for n in imp.history if self._in_table(n)]
            for pos in clicked:
                if not self._in_table(pos):
                    continue
                for _ in range(negatives_per_pos):
                    neg = self._rng.choice(non_clicked)
                    self.triplets.append((history, pos, neg))

    def _in_table(self, nid: str) -> bool:
        has = getattr(self.news_tokens, "has", None)
        return has(nid) if has else (nid in self.news_tokens)

    def _pop(self, nid: str) -> float:
        if self.popularity is None:
            return 0.0
        return float(self.popularity.count(nid))

    def __len__(self) -> int:
        return len(self.triplets)

    # ------------------------------------------------------------------ #
    def _news_tensor(self, nid: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.news_tokens.get(nid)
        return (
            torch.tensor(tok["input_ids"], dtype=torch.long),
            torch.tensor(tok["attention_mask"], dtype=torch.long),
        )

    def _history_tensors(self, history: List[str]):
        hist = history[-self.max_history:]
        ids = torch.zeros(self.max_history, self.max_len, dtype=torch.long)
        attn = torch.zeros(self.max_history, self.max_len, dtype=torch.long)
        mask = torch.zeros(self.max_history, dtype=torch.float)
        pop = torch.zeros(self.max_history, dtype=torch.float)
        for i, nid in enumerate(hist):
            tok = self.news_tokens.get(nid)
            ids[i] = torch.tensor(tok["input_ids"], dtype=torch.long)
            attn[i] = torch.tensor(tok["attention_mask"], dtype=torch.long)
            mask[i] = 1.0
            pop[i] = self._pop(nid)
        return ids, attn, mask, pop

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        history, pos, neg = self.triplets[index]
        hist_ids, hist_attn, hist_mask, hist_pop = self._history_tensors(history)
        pos_ids, pos_attn = self._news_tensor(pos)
        neg_ids, neg_attn = self._news_tensor(neg)
        return {
            "history_input_ids": hist_ids,
            "history_attention_mask": hist_attn,
            "history_mask": hist_mask,
            "history_pop": hist_pop,
            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_attn,
            "neg_input_ids": neg_ids,
            "neg_attention_mask": neg_attn,
            "pos_pop": torch.tensor(self._pop(pos), dtype=torch.float),
            "neg_pop": torch.tensor(self._pop(neg), dtype=torch.float),
        }
