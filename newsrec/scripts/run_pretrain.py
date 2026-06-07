"""
run_pretrain.py
===============

Config-driven pre-training entry point.

Usage::

    python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/full.yaml \\
        [key=value ...]
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.pretrain_dataset import PretrainDataset, build_user_sequences
from newsrec.losses.pretrain_losses import select_enabled_tasks, task_weights
from newsrec.models.pretrain_model import PretrainModule
from newsrec.models.rec_model import build_rec_model
from newsrec.scripts.common import (
    build_checkpoint_manager,
    build_logger,
    load_news_and_tokens,
)
from newsrec.training.pretrainer import Pretrainer
from newsrec.utils.config import load_config
from newsrec.utils.env import load_dotenv
from newsrec.utils.seed import set_seed


def run_pretrain(cfg):
    load_dotenv(cfg.get("env_file", ".env"))
    set_seed(int(cfg.get("seed", 42)))
    logger = build_logger(cfg, "pretrain")
    device = cfg.get("device", "cpu")

    train, dev, news_tokens, category_ids, cat_vocab, popularity, encoder = load_news_and_tokens(cfg)

    sequences = build_user_sequences(
        train.impressions,
        min_len=int(cfg.get("data.min_seq_len", 3)),
        in_table=news_tokens.has,
    )
    dataset = PretrainDataset(
        sequences, news_tokens, category_ids,
        max_seq_len=int(cfg.get("data.max_history", 50)),
        mask_prob=float(cfg.get("data.mask_prob", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    logger.info(f"Pretrain sequences: {len(dataset)}")

    enabled = select_enabled_tasks(cfg.get("pretrain.tasks", ["aap", "mip", "map", "sp", "bsm"]))
    weights = task_weights(cfg.get("pretrain.tasks"), enabled)
    logger.info(f"Enabled pretrain tasks: {enabled} weights={weights}")

    model = build_rec_model(cfg.get("model", {}).to_dict() if cfg.get("model") else {})
    module = PretrainModule(
        model, num_categories=len(cat_vocab), enabled_tasks=enabled,
        weights=weights, tau=float(cfg.get("pretrain.tau", 0.1)),
    )

    ckpt = build_checkpoint_manager(cfg, "pretrain", logger, tokenizer=encoder.tokenizer)
    train_cfg = cfg.get("pretrain.training", {})
    train_cfg = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
    loader = DataLoader(
        dataset, batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True, collate_fn=stack_collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    trainer = Pretrainer(module, config=train_cfg, device=device, logger=logger,
                         checkpoint_manager=ckpt)
    trainer.train(loader)
    ckpt.close()
    logger.info("Pre-training complete.")


def main():
    parser = argparse.ArgumentParser(description="Pre-train the news recommender.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, cli_overrides=args.overrides)
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
