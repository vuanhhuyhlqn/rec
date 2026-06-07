"""
download.py
===========

Auto-download MIND splits from a HuggingFace *dataset* repo when they are not
present locally — so the same config runs unchanged on a fresh machine.

The dataset repo is expected to store each split under its own subfolder, e.g.::

    <repo_id>/
        train/  news.tsv  behaviors.tsv  popularity_data.csv
        dev/    news.tsv  behaviors.tsv

:func:`ensure_mind_split` returns a local directory that contains ``news.tsv``
(+ ``behaviors.tsv``): the original ``local_dir`` if it already has the data,
otherwise a snapshot downloaded from the Hub.
"""

from __future__ import annotations

import os
from typing import Optional


def _has_split(path: str) -> bool:
    return os.path.exists(os.path.join(path, "news.tsv"))


def ensure_mind_split(
    local_dir: str,
    repo_id: Optional[str] = None,
    subfolder: str = "train",
    auto_download: bool = False,
    token: Optional[str] = None,
    logger=None,
    download_dir: str = "dataset",
) -> str:
    """
    Resolve a usable local path for a MIND split.

    Order of preference:
    1. ``local_dir`` if it already contains ``news.tsv``.
    2. ``{download_dir}/{subfolder}`` if a previous download already populated it.
    3. Download ``{subfolder}/*`` from the HF dataset ``repo_id`` (when
       ``auto_download`` is set) **into** ``download_dir`` and return
       ``{download_dir}/{subfolder}``.

    Unlike the default ``snapshot_download`` behaviour (which stores files in
    the global HF cache, e.g. ``~/.cache/huggingface/hub``), this places the
    data in a project-local ``download_dir`` (default ``dataset/``).
    """
    if _has_split(local_dir):
        return local_dir

    # Reuse a prior download under download_dir/<subfolder> if present.
    cached = os.path.join(download_dir, subfolder)
    if _has_split(cached):
        return cached

    if not (auto_download and repo_id):
        # Nothing we can do; let the caller raise a clear FileNotFoundError.
        return local_dir

    from huggingface_hub import snapshot_download

    os.makedirs(download_dir, exist_ok=True)
    if logger:
        logger.info(
            f"Downloading MIND '{subfolder}' split from dataset repo {repo_id} "
            f"into {download_dir}/ ..."
        )
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{subfolder}/*"],
        token=token,
        local_dir=download_dir,
    )
    resolved = os.path.join(download_dir, subfolder)
    if not _has_split(resolved):
        # Some repos store files at the root rather than in a subfolder.
        if _has_split(snapshot):
            resolved = snapshot
    if logger:
        logger.info(f"MIND '{subfolder}' split ready at {resolved}")
    return resolved
