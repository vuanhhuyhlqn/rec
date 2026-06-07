"""
common.py
=========

Shared builders used by the ``run_pretrain`` / ``run_finetune`` entry points:
loading MIND data + tokenising the news catalogue, building popularity, and
wiring the logger + (async-HF) checkpoint manager from a config.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from newsrec.data.download import ensure_mind_split
from newsrec.data.mind_parser import load_mind_split
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.data.popularity import ItemPopularity
from newsrec.data.vocab import NewsTextEncoder, build_category_vocab
from newsrec.training.checkpoint import CheckpointManager
from newsrec.training.hub_uploader import HubUploader
from newsrec.utils.env import get_hf_token
from newsrec.utils.logging import setup_logger


def build_logger(cfg, stage: str):
    log_dir = cfg.get("logging.log_dir", "logs")
    run_name = cfg.get("run_name", "default")
    level = cfg.get("logging.level", "INFO")
    return setup_logger(name=f"newsrec.{stage}", log_dir=log_dir, stage=stage,
                        run_name=run_name, level=level)


def resolve_device(cfg, logger=None) -> str:
    """
    Resolve the training device from config.

    * ``auto`` (or unset) -> ``cuda`` if a GPU is visible, else ``cpu``.
    * ``cuda`` requested but no GPU available -> fall back to ``cpu`` with a
      warning (prevents silent CPU OOM when a run *expected* a GPU).
    * any explicit value is honoured otherwise.
    """
    requested = str(cfg.get("device", "auto")).lower()
    has_cuda = torch.cuda.is_available()
    if requested in ("auto", "", "none"):
        device = "cuda" if has_cuda else "cpu"
    elif requested == "cuda" and not has_cuda:
        if logger:
            logger.warning("device=cuda requested but no GPU is visible; falling back to cpu")
        device = "cpu"
    else:
        device = requested
    if logger:
        logger.info(f"Using device: {device}"
                    + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    return device


def resolve_batch_size(configured, dataset, compute_losses, optimizer, model,
                       device, cfg, logger=None) -> int:
    """
    Return an integer batch size.

    If ``configured`` is the string ``"auto"`` (or ``cfg['auto_batch_size']`` is
    truthy), probe the GPU for the largest batch size that fits (see
    :func:`newsrec.training.batch_finder.find_max_batch_size`). On non-CUDA
    devices auto-sizing is not possible, so a ``fallback_batch_size`` is used.

    Relevant ``cfg`` keys: ``max_batch_size`` (search cap, default 256),
    ``batch_safety`` (headroom multiplier, default 0.9),
    ``fallback_batch_size`` (CPU fallback, default 8).
    """
    auto = (isinstance(configured, str) and configured.lower() == "auto") \
        or bool(cfg.get("auto_batch_size", False))
    if not auto:
        return int(configured)

    from newsrec.data.collate import stack_collate
    from newsrec.training.batch_finder import find_max_batch_size

    def _build(bs):
        n = len(dataset)
        return stack_collate([dataset[i % n] for i in range(bs)])

    def _probe(batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = compute_losses(batch)
        out["total"].backward()
        optimizer.zero_grad(set_to_none=True)

    cap = max(1, min(int(cfg.get("max_batch_size", 256)), len(dataset)))
    found = find_max_batch_size(
        _build, _probe, device=device, max_batch=cap,
        safety=float(cfg.get("batch_safety", 0.9)), logger=logger,
    )
    if found is None:
        fallback = int(cfg.get("fallback_batch_size", 8))
        if logger:
            logger.info(f"auto batch size requested but device is not CUDA; "
                        f"using fallback batch_size={fallback}")
        return fallback
    if logger:
        logger.info(f"auto-selected batch_size={found}")
    return found


def build_checkpoint_manager(cfg, stage: str, logger, tokenizer=None):
    root = cfg.get("checkpoint.dir", "checkpoints")
    run_name = cfg.get("run_name", "run")
    ckpt_dir = os.path.join(root, run_name, stage)
    token = get_hf_token()
    push = bool(cfg.get("hub.push_to_hub", False))
    if push and logger is not None:
        logger.info(f"HF push enabled; token detected: {bool(token)}; repo: {cfg.get('hub.hub_repo_id')}")
    uploader = HubUploader(
        repo_id=cfg.get("hub.hub_repo_id"),
        token=token,
        private=bool(cfg.get("hub.hub_private", False)),
        enabled=push,
        logger=logger,
    )
    return CheckpointManager(ckpt_dir, config=cfg.to_dict(), uploader=uploader,
                             logger=logger, tokenizer=tokenizer)


def load_news_and_tokens(cfg) -> Tuple:
    """Load train/dev MIND, tokenise the union of news, build category ids + popularity.

    Honours optional ``data.max_train_impressions`` / ``data.max_dev_impressions``
    limits (used by the smoke test): when set, only those impressions are kept
    and only the news they reference are tokenised, keeping the run tiny.
    """
    train_dir = cfg.get("data.train_dir", "dataset/train")
    dev_dir = cfg.get("data.dev_dir", "dataset/dev")
    max_title_len = int(cfg.get("data.max_title_len", 64))
    plm_name = cfg.get("model.plm.model_name", "bert-base-uncased")
    max_train = cfg.get("data.max_train_impressions")
    max_dev = cfg.get("data.max_dev_impressions")

    # Auto-download from a HF dataset repo when the local split is absent.
    repo_id = cfg.get("data.hf_dataset_repo")
    auto_download = bool(cfg.get("data.auto_download", False))
    download_dir = cfg.get("data.download_dir", "dataset")
    token = get_hf_token()
    train_dir = ensure_mind_split(train_dir, repo_id, cfg.get("data.train_subfolder", "train"),
                                  auto_download, token, download_dir=download_dir)
    dev_dir = ensure_mind_split(dev_dir, repo_id, cfg.get("data.dev_subfolder", "dev"),
                                auto_download, token, download_dir=download_dir)

    train = load_mind_split(train_dir)
    dev = load_mind_split(dev_dir) if os.path.isdir(dev_dir) else None

    if max_train:
        train.impressions = train.impressions[: int(max_train)]
    if dev is not None and max_dev:
        dev.impressions = dev.impressions[: int(max_dev)]

    def _referenced(data):
        nids = set()
        for imp in data.impressions:
            nids.update(imp.history)
            nids.update(n for n, _ in imp.candidates)
        return nids

    if max_train or max_dev:
        ref = _referenced(train)
        if dev is not None:
            ref |= _referenced(dev)
        all_news = {nid: train.news[nid] for nid in ref if nid in train.news}
        if dev is not None:
            for nid in ref:
                if nid not in all_news and nid in dev.news:
                    all_news[nid] = dev.news[nid]
    else:
        all_news = dict(train.news)
        if dev is not None:
            all_news.update(dev.news)

    encoder = NewsTextEncoder(plm_name, max_len=max_title_len)
    news_tokens = NewsTokenTable.build(all_news, encoder)

    cat_vocab = build_category_vocab(train.news, *( [dev.news] if dev else [] ))
    category_ids = {nid: cat_vocab.index(item.category) for nid, item in all_news.items()}

    popularity = ItemPopularity.from_impressions(train.impressions)

    return train, dev, news_tokens, category_ids, cat_vocab, popularity, encoder
