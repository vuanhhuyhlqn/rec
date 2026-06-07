"""
run_finetune.py
===============

Config-driven PAAC fine-tuning entry point.

Usage::

    python -m newsrec.scripts.run_finetune --config newsrec/config/finetune/paac.yaml \\
        [key=value ...]
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.finetune_dataset import FinetuneTripletDataset
from newsrec.models.rec_model import build_rec_model
from newsrec.scripts.common import (
    build_checkpoint_manager,
    build_logger,
    load_news_and_tokens,
    resolve_batch_size,
    resolve_device,
)
from newsrec.training.finetuner import Finetuner
from newsrec.utils.config import load_config
from newsrec.utils.env import load_dotenv
from newsrec.utils.seed import set_seed


def run_finetune(cfg):
    load_dotenv(cfg.get("env_file", ".env"))
    set_seed(int(cfg.get("seed", 42)))
    logger = build_logger(cfg, "finetune")
    device = resolve_device(cfg, logger)

    train, dev, news_tokens, category_ids, cat_vocab, popularity, encoder = load_news_and_tokens(cfg)

    dataset = FinetuneTripletDataset(
        train.impressions, news_tokens,
        max_history=int(cfg.get("data.max_history", 50)),
        negatives_per_pos=int(cfg.get("finetune.negatives_per_pos", 1)),
        seed=int(cfg.get("seed", 42)),
        popularity=popularity,
        resample_negatives=bool(cfg.get("finetune.resample_negatives", True)),
    )
    logger.info(f"Finetune triplets: {len(dataset)}")

    model = build_rec_model(cfg.get("model", {}).to_dict() if cfg.get("model") else {})
    # Training is LoRA-only from the start: the PLM base weights stay frozen and
    # only the LoRA adapters (+ the encoders/heads) train. No gradual unfreeze.

    ckpt = build_checkpoint_manager(cfg, "finetune", logger, tokenizer=encoder.tokenizer)

    ft_cfg = cfg.get("finetune", {})
    ft_cfg = ft_cfg.to_dict() if hasattr(ft_cfg, "to_dict") else dict(ft_cfg)
    # only pass the trainer-relevant keys
    trainer_cfg = {k: v for k, v in ft_cfg.items()
                   if k not in ("pretrained_ckpt", "negatives_per_pos",
                                "batch_size", "num_workers", "resample_negatives")}
    trainer_cfg["max_history"] = int(cfg.get("data.max_history", 50))

    tuner = Finetuner(model, config=trainer_cfg, device=device, logger=logger,
                      popularity=popularity, checkpoint_manager=ckpt)

    pretrained = cfg.get("finetune.pretrained_ckpt")
    if pretrained:
        tuner.load_pretrained(pretrained)

    batch_size = resolve_batch_size(
        ft_cfg.get("batch_size", 16), dataset, tuner.compute_losses, tuner.optimizer,
        tuner.model, device, ft_cfg, logger,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=stack_collate,
        num_workers=int(ft_cfg.get("num_workers", 0)),
    )

    dev_impressions = dev.impressions if dev is not None else None
    tuner.train(loader, dev_impressions=dev_impressions, news_tokens=news_tokens)
    ckpt.close()
    logger.info("Fine-tuning complete.")


def main():
    parser = argparse.ArgumentParser(description="PAAC fine-tune the news recommender.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, cli_overrides=args.overrides)
    run_finetune(cfg)


if __name__ == "__main__":
    main()
