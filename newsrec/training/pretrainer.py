"""
pretrainer.py
=============

Joint multi-task pre-training loop over a :class:`PretrainModule`.

Runs the enabled self-supervised tasks (AAP / MIP / MAP / SP / BSM) with their
configured weights, logs the per-task loss breakdown to the run log file, and
periodically checkpoints (locally + async HF) via a
:class:`~newsrec.training.checkpoint.CheckpointManager`.  The ``best/``
checkpoint tracks the lowest running pre-training loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import contextlib

import torch
from torch.utils.data import DataLoader

try:  # progress bars are optional; degrade gracefully if tqdm is absent
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from newsrec.utils.logging import format_metrics


DEFAULT_PRETRAIN_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "epochs": 1,
    "log_every": 10,
    "progress": True,
    "save_every": 1,  # epochs
}


class Pretrainer:
    def __init__(
        self,
        pretrain_module,
        config: Optional[dict] = None,
        device: str | torch.device = "cpu",
        logger=None,
        checkpoint_manager=None,
    ):
        self.module = pretrain_module
        self.cfg = dict(DEFAULT_PRETRAIN_CONFIG)
        if config:
            self.cfg.update(config)
        self.device = torch.device(device)
        self.module.to(self.device)
        self.logger = logger
        self.ckpt = checkpoint_manager

        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        self.global_step = 0

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _log_step(self, msg: str) -> None:
        # Per-step lines go to the log file only (DEBUG) so they don't break the
        # console tqdm progress bar.
        if self.logger:
            self.logger.debug(msg)

    # ------------------------------------------------------------------ #
    def _autocast(self):
        # bf16 mixed precision on GPU (no GradScaler needed for bf16).
        if self.cfg.get("amp", True) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def train_step(self, batch) -> Dict[str, float]:
        self.module.train()
        with self._autocast():
            losses = self.module.compute_losses(batch)
        self.optimizer.zero_grad()
        losses["total"].backward()
        if self.cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()
        self.global_step += 1
        return {k: float(v.detach()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    def train(self, train_loader: DataLoader, epochs: Optional[int] = None) -> Dict[str, float]:
        epochs = epochs or self.cfg["epochs"]
        last: Dict[str, float] = {}
        for epoch in range(epochs):
            running = 0.0
            use_bar = tqdm is not None and self.cfg.get("progress", True)
            bar = tqdm(total=len(train_loader), desc=f"pretrain e{epoch}",
                       leave=False, dynamic_ncols=True) if use_bar else None
            for i, batch in enumerate(train_loader):
                step_losses = self.train_step(batch)
                running += step_losses["total"]
                last = step_losses
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(loss=f"{step_losses['total']:.4f}",
                                    avg=f"{running / (i + 1):.4f}")
                if (i + 1) % self.cfg["log_every"] == 0:
                    self._log_step(format_metrics(step_losses, prefix=f"[pretrain e{epoch} s{i + 1}/{len(train_loader)}]"))
            if bar is not None:
                bar.close()

            avg = running / max(1, len(train_loader))
            self._log(f"[pretrain e{epoch}] avg_total_loss={avg:.4f}")

            if self.ckpt is not None and (epoch + 1) % self.cfg["save_every"] == 0:
                # Lower loss is better for the pretrain 'best' checkpoint.
                self.ckpt.save(
                    self.module.model,
                    tag=f"epoch{epoch}",
                    extra_state={
                        "category_embeddings": self.module.category_embeddings.state_dict(),
                        "mask_token": self.module.mask_token.detach().cpu(),
                    },
                    metric=avg,
                    higher_is_better=False,
                )
        return last
