"""
finetuner.py
============

Fine-tuning trainer.

Task 8 implements the **L_rec-only** training loop (BPR over in-impression
triplets) together with:

* periodic dev evaluation via :class:`~newsrec.eval.evaluator.ImpressionEvaluator`,
* file + console logging of the per-step / per-epoch loss and dev metrics,
* the :class:`~newsrec.training.lora_schedule.LoRAUnfreezeScheduler` hook.

:meth:`Finetuner.compute_losses` is written so Task 9 can extend it with the
PAAC ``L_sa`` / ``L_cl`` terms without touching the training loop.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch

try:  # progress bars are optional; degrade gracefully if tqdm is absent
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None
from torch.utils.data import DataLoader

from newsrec.eval.evaluator import ImpressionEvaluator
from newsrec.losses.paac_losses import (
    bpr_loss,
    l2_regularization,
    reweighting_contrastive_loss,
    supervised_alignment_loss,
)
from newsrec.utils.logging import format_metrics


DEFAULT_FINETUNE_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "epochs": 1,
    "log_every": 10,
    "progress": True,
    "eval_every": 1,
    "max_eval_impressions": 2000,
    "max_history": 50,
    # PAAC multi-task weights (Task 9 uses lambda1 / lambda2).
    "lambda1": 0.0,   # L_sa weight
    "lambda2": 0.0,   # L_cl weight
    "lambda3": 1e-4,  # L2 regularisation weight
    # PAAC loss hyper-parameters.
    "sa_ratio": 0.5,      # per-user popular fraction for L_sa
    "cl_x_percent": 50.0, # batch-level top-x% popular split for L_cl
    "cl_beta": 1.0,       # weight of cross-group negatives in L_cl
    "cl_gamma": 0.5,      # balance between pop-anchor and unpop-anchor L_cl
    "cl_tau": 0.1,        # InfoNCE temperature
    "cl_dropout": 0.1,    # augmentation feature dropout
    "cl_noise": 0.1,      # augmentation gaussian noise std
}


class Finetuner:
    def __init__(
        self,
        model,
        config: Optional[dict] = None,
        device: str | torch.device = "cpu",
        logger=None,
        scheduler=None,
        popularity=None,
        checkpoint_manager=None,
    ):
        self.model = model
        self.cfg = dict(DEFAULT_FINETUNE_CONFIG)
        if config:
            self.cfg.update(config)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.logger = logger
        self.scheduler = scheduler
        self.popularity = popularity
        self.ckpt = checkpoint_manager

        # Optimiser over ALL params so gradual unfreeze works without rebuild.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        self.evaluator = ImpressionEvaluator(self.model, device=self.device)
        self.global_step = 0

    # ------------------------------------------------------------------ #
    def load_pretrained(self, ckpt_dir_or_file: str):
        """Load pre-trained Module 0/1/2 weights into the model (non-strict)."""
        from newsrec.training.checkpoint import load_backbone_weights

        result = load_backbone_weights(self.model, ckpt_dir_or_file, strict=False)
        missing, unexpected = result
        self._log(
            f"Loaded pretrained backbone from {ckpt_dir_or_file} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return result

    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _log_step(self, msg: str) -> None:
        # Per-step lines go to the log file only (DEBUG) so they don't break the
        # console tqdm progress bar.
        if self.logger:
            self.logger.debug(msg)

    # ------------------------------------------------------------------ #
    # Forward / losses                                                   #
    # ------------------------------------------------------------------ #
    def _encode_triplet(self, batch: Dict[str, torch.Tensor]):
        # Module-1 news embeddings of the history (needed by L_sa), then the
        # user encoder (Module 2) produces z_u.
        history_vecs = self.model.encode_news(
            batch["history_input_ids"].to(self.device),
            batch["history_attention_mask"].to(self.device),
        )  # [B, S, D]
        history_mask = batch["history_mask"].to(self.device)
        _, z_u = self.model.user_encoder(history_vecs, history_mask)
        pos_vec = self.model.encode_news(
            batch["pos_input_ids"].to(self.device),
            batch["pos_attention_mask"].to(self.device),
        )
        neg_vec = self.model.encode_news(
            batch["neg_input_ids"].to(self.device),
            batch["neg_attention_mask"].to(self.device),
        )
        return z_u, pos_vec, neg_vec, history_vecs, history_mask

    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return a dict of loss components including ``total``.

        Implements the PAAC multi-task objective
        ``L_total = L_rec + l1*L_sa + l2*L_cl + l3*||Theta||^2``.
        """
        z_u, pos_vec, neg_vec, history_vecs, history_mask = self._encode_triplet(batch)
        pos_score = self.model.score(z_u, pos_vec)  # [B]
        neg_score = self.model.score(z_u, neg_vec)  # [B]
        l_rec = bpr_loss(pos_score, neg_score)

        losses: Dict[str, torch.Tensor] = {"L_rec": l_rec}
        total = l_rec

        # L_sa — popularity-aware supervised alignment (user-level)
        if self.cfg["lambda1"] and "history_pop" in batch:
            l_sa = supervised_alignment_loss(
                history_vecs,
                history_mask,
                batch["history_pop"].to(self.device),
                ratio=self.cfg["sa_ratio"],
            )
            losses["L_sa"] = l_sa
            total = total + self.cfg["lambda1"] * l_sa

        # L_cl — re-weighting contrastive learning (batch-level on positives)
        if self.cfg["lambda2"] and "pos_pop" in batch:
            l_cl = reweighting_contrastive_loss(
                pos_vec,
                batch["pos_pop"].to(self.device),
                x_percent=self.cfg["cl_x_percent"],
                beta=self.cfg["cl_beta"],
                gamma=self.cfg["cl_gamma"],
                tau=self.cfg["cl_tau"],
                dropout_p=self.cfg["cl_dropout"],
                noise_std=self.cfg["cl_noise"],
            )
            losses["L_cl"] = l_cl
            total = total + self.cfg["lambda2"] * l_cl

        if self.cfg["lambda3"]:
            reg = l2_regularization(self.model.parameters()).to(total.device)
            losses["L_reg"] = reg
            total = total + self.cfg["lambda3"] * reg

        losses["total"] = total
        return losses

    def _autocast(self):
        # bf16 mixed precision on GPU: ~1.5-2x faster + lower memory (so the
        # auto batch-finder can pick a larger batch). bf16 needs no GradScaler.
        if self.cfg.get("amp", True) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ #
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        with self._autocast():
            losses = self.compute_losses(batch)
        self.optimizer.zero_grad()
        losses["total"].backward()
        if self.cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()
        self.global_step += 1
        return {k: float(v.detach()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    def train(
        self,
        train_loader: DataLoader,
        dev_impressions=None,
        news_tokens=None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        epochs = epochs or self.cfg["epochs"]
        last_metrics: Dict[str, float] = {}

        for epoch in range(epochs):
            if self.scheduler is not None:
                changed, n_layers = self.scheduler.step(epoch)
                if changed:
                    self._log(f"[epoch {epoch}] LoRA unfreeze -> top {n_layers} BERT layers")

            running = 0.0
            use_bar = tqdm is not None and self.cfg.get("progress", True)
            bar = tqdm(total=len(train_loader), desc=f"finetune e{epoch}",
                       leave=False, dynamic_ncols=True) if use_bar else None
            for i, batch in enumerate(train_loader):
                step_losses = self.train_step(batch)
                running += step_losses["total"]
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(loss=f"{step_losses['total']:.4f}",
                                    avg=f"{running / (i + 1):.4f}")
                if (i + 1) % self.cfg["log_every"] == 0:
                    self._log_step(format_metrics(step_losses, prefix=f"[epoch {epoch} step {i + 1}/{len(train_loader)}]"))
            if bar is not None:
                bar.close()

            avg = running / max(1, len(train_loader))
            self._log(f"[epoch {epoch}] avg_total_loss={avg:.4f}")

            if dev_impressions is not None and news_tokens is not None and (
                (epoch + 1) % self.cfg["eval_every"] == 0
            ):
                last_metrics = self.evaluate(dev_impressions, news_tokens)
                self._log(format_metrics(last_metrics, prefix=f"[epoch {epoch} DEV]"))
                if self.ckpt is not None:
                    auc = last_metrics.get("auc", float("nan"))
                    metric = None if auc != auc else auc  # skip NaN
                    self.ckpt.save(self.model, tag=f"epoch{epoch}", metric=metric,
                                   higher_is_better=True)
            elif self.ckpt is not None and (epoch + 1) % self.cfg["eval_every"] == 0:
                self.ckpt.save(self.model, tag=f"epoch{epoch}")

        return last_metrics

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, dev_impressions, news_tokens) -> Dict[str, float]:
        news_vectors = self.evaluator.encode_news_table(
            {nid: news_tokens.get(nid) for nid in _unique_news(dev_impressions, news_tokens)}
        )
        capacity = self.model.user_encoder.fastformer.position_embeddings.num_embeddings
        max_history = min(self.cfg.get("max_history", 50), capacity)
        return self.evaluator.evaluate(
            dev_impressions,
            news_vectors,
            max_history=max_history,
            max_impressions=self.cfg["max_eval_impressions"],
        )


def _unique_news(impressions, news_tokens) -> set:
    nids = set()
    for imp in impressions:
        nids.update(imp.history)
        nids.update(nid for nid, _ in imp.candidates)
    return {n for n in nids if (news_tokens.has(n) if hasattr(news_tokens, "has") else n in news_tokens)}
