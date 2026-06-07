"""
checkpoint.py
=============

Local checkpoint save / load + a :class:`CheckpointManager` that combines local
saving with asynchronous HuggingFace uploads (periodic + best-by-metric).

A checkpoint directory contains:

* ``model.pt``        — ``NewsRecModel`` state dict (Modules 0/1/2 + pooler).
* ``config.yaml``     — the run config (if provided).
* ``lora/``           — LoRA adapters (peft ``save_pretrained``) if present.
* ``tokenizer/``      — the BERT tokenizer (if provided).
* ``extra.pt``        — optional extra state (e.g. pre-training heads).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from newsrec.utils.config import save_config


def save_checkpoint(
    model,
    out_dir: str,
    config=None,
    tokenizer=None,
    extra_state: Optional[dict] = None,
    save_lora: bool = True,
) -> str:
    """Save a checkpoint locally; returns ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    if config is not None:
        save_config(config, os.path.join(out_dir, "config.yaml"))

    if extra_state is not None:
        torch.save(extra_state, os.path.join(out_dir, "extra.pt"))

    # LoRA adapters (peft) — model.plm.bert is a PeftModel when LoRA is on.
    if save_lora:
        bert = getattr(getattr(model, "plm", None), "bert", None)
        if bert is not None and hasattr(bert, "save_pretrained") and hasattr(bert, "peft_config"):
            try:
                bert.save_pretrained(os.path.join(out_dir, "lora"))
            except Exception:  # pragma: no cover - best effort
                pass

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))

    return out_dir


def load_backbone_weights(model, ckpt_dir_or_file: str, strict: bool = False):
    """
    Load the ``NewsRecModel`` backbone weights (Modules 0/1/2) from a checkpoint.

    Accepts either a directory containing ``model.pt`` or the file directly.
    Returns the ``(missing_keys, unexpected_keys)`` from ``load_state_dict``.
    """
    path = ckpt_dir_or_file
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    state = torch.load(path, map_location="cpu")
    return model.load_state_dict(state, strict=strict)


class CheckpointManager:
    """Saves checkpoints locally and (optionally) pushes them to the Hub async."""

    def __init__(self, local_dir: str, config=None, uploader=None, logger=None, tokenizer=None):
        self.local_dir = local_dir
        self.config = config
        self.uploader = uploader
        self.logger = logger
        self.tokenizer = tokenizer
        self.best_metric: Optional[float] = None
        os.makedirs(local_dir, exist_ok=True)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def save(self, model, tag: str, extra_state=None, metric: Optional[float] = None,
             higher_is_better: bool = True) -> str:
        """Save a periodic checkpoint under ``{local_dir}/{tag}`` and push it."""
        out_dir = os.path.join(self.local_dir, tag)
        save_checkpoint(model, out_dir, config=self.config, tokenizer=self.tokenizer,
                        extra_state=extra_state)
        self._log(f"Saved checkpoint: {out_dir}")
        if self.uploader is not None:
            self.uploader.enqueue(out_dir, tag)

        if metric is not None:
            self.maybe_save_best(model, metric, extra_state, higher_is_better)
        return out_dir

    def maybe_save_best(self, model, metric: float, extra_state=None,
                        higher_is_better: bool = True) -> bool:
        """Save a separate ``best/`` checkpoint when ``metric`` improves."""
        improved = (
            self.best_metric is None
            or (metric > self.best_metric if higher_is_better else metric < self.best_metric)
        )
        if improved:
            self.best_metric = metric
            out_dir = os.path.join(self.local_dir, "best")
            save_checkpoint(model, out_dir, config=self.config, tokenizer=self.tokenizer,
                            extra_state=extra_state)
            self._log(f"New best ({metric:.4f}); saved {out_dir}")
            if self.uploader is not None:
                self.uploader.enqueue(out_dir, "best")
        return improved

    def close(self) -> None:
        if self.uploader is not None:
            self.uploader.close(wait=True)
