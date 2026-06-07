"""
evaluator.py
============

Ranks the candidates of each impression with a :class:`NewsRecModel` and
reports the averaged MIND metrics.

Two-phase design (mirrors the representation caching in Legommenders):

1. :meth:`encode_news_table` runs the (expensive) PLM + news encoder **once**
   over the unique news, producing a ``{news_id: vector}`` table.
2. :meth:`evaluate` reuses those cached vectors: it only runs the (cheap) user
   encoder + cosine scoring per impression, in mini-batches.

The scoring path is intentionally decoupled from BERT so it can be unit-tested
with arbitrary injected vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from newsrec.eval.metrics import compute_impression_metrics


class ImpressionEvaluator:
    def __init__(self, model, device: str | torch.device = "cpu"):
        self.model = model
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Phase 1: encode the news catalogue                                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_news_table(
        self,
        news_tokens: Dict[str, Dict[str, Sequence[int]]],
        batch_size: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """``{nid: {input_ids, attention_mask}}`` → ``{nid: vector [D]}``."""
        self.model.eval()
        nids = list(news_tokens.keys())
        vectors: Dict[str, torch.Tensor] = {}
        for start in range(0, len(nids), batch_size):
            chunk = nids[start:start + batch_size]
            ids = torch.tensor(
                [news_tokens[n]["input_ids"] for n in chunk], device=self.device
            )
            attn = torch.tensor(
                [news_tokens[n]["attention_mask"] for n in chunk], device=self.device
            )
            vecs = self.model.encode_news(ids, attn)  # [chunk, D]
            for nid, vec in zip(chunk, vecs):
                vectors[nid] = vec.detach().cpu()
        return vectors

    # ------------------------------------------------------------------ #
    # Phase 2: rank impressions                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(
        self,
        impressions: Sequence,
        news_vectors: Dict[str, torch.Tensor],
        max_history: int = 50,
        batch_size: int = 64,
        max_impressions: Optional[int] = None,
        metrics: Sequence[str] = ("auc", "mrr", "ndcg@5", "ndcg@10"),
    ) -> Dict[str, float]:
        self.model.eval()
        dim = next(iter(news_vectors.values())).shape[-1]
        zero = torch.zeros(dim)

        # Keep only impressions with at least one candidate.
        imprs = [imp for imp in impressions if imp.candidates]
        if max_impressions is not None:
            imprs = imprs[:max_impressions]

        scores_list: List[List[float]] = []
        labels_list: List[List[int]] = []

        for start in range(0, len(imprs), batch_size):
            batch = imprs[start:start + batch_size]
            scores_batch, labels_batch = self._score_batch(
                batch, news_vectors, zero, max_history
            )
            scores_list.extend(scores_batch)
            labels_list.extend(labels_batch)

        return compute_impression_metrics(scores_list, labels_list, metrics)

    # ------------------------------------------------------------------ #
    def _lookup(self, nid: str, news_vectors, zero) -> torch.Tensor:
        return news_vectors.get(nid, zero)

    @torch.no_grad()
    def _score_batch(self, batch, news_vectors, zero, max_history):
        max_hist = max(1, min(max_history, max((len(imp.history) for imp in batch), default=1)))
        max_cand = max(len(imp.candidates) for imp in batch)
        B = len(batch)
        dim = zero.shape[-1]

        hist = torch.zeros(B, max_hist, dim)
        hist_mask = torch.zeros(B, max_hist)
        cand = torch.zeros(B, max_cand, dim)
        cand_mask = torch.zeros(B, max_cand, dtype=torch.bool)
        labels = torch.zeros(B, max_cand, dtype=torch.long)

        for b, imp in enumerate(batch):
            history = imp.history[-max_hist:]
            for s, nid in enumerate(history):
                hist[b, s] = self._lookup(nid, news_vectors, zero)
                hist_mask[b, s] = 1.0
            for k, (nid, label) in enumerate(imp.candidates):
                cand[b, k] = self._lookup(nid, news_vectors, zero)
                cand_mask[b, k] = True
                labels[b, k] = label

        hist = hist.to(self.device)
        hist_mask = hist_mask.to(self.device)
        cand = cand.to(self.device)

        _, z_u = self.model.user_encoder(hist, hist_mask)
        scores = self.model.score(z_u, cand).cpu()  # [B, max_cand]

        scores_out: List[List[float]] = []
        labels_out: List[List[int]] = []
        for b in range(B):
            valid = cand_mask[b]
            scores_out.append(scores[b][valid].tolist())
            labels_out.append(labels[b][valid].tolist())
        return scores_out, labels_out
