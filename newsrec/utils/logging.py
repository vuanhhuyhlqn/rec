"""
logging.py
==========

Console + per-run file logging for the training pipeline.

A single :func:`setup_logger` call wires a logger that emits to both stdout and
a timestamped file ``{log_dir}/{stage}_{run_name}_{timestamp}.log``.

Trainers additionally use :func:`format_metrics` to render a tidy single-line
summary of a loss / metric dictionary for the log.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Mapping


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_log_path(log_dir: str, stage: str, run_name: str) -> str:
    """Return ``{log_dir}/{stage}_{run_name}_{timestamp}.log`` (dir created)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stage}_{run_name}_{timestamp}.log"
    return os.path.join(log_dir, filename)


def setup_logger(
    name: str = "newsrec",
    log_dir: str = "logs",
    stage: str = "run",
    run_name: str = "default",
    level: str | int = "INFO",
    log_path: str | None = None,
    to_console: bool = True,
) -> logging.Logger:
    """
    Create (or reconfigure) a logger that writes to console and a file.

    Returns the logger; the resolved file path is stored on
    ``logger.log_path`` for convenience.
    """
    if log_path is None:
        log_path = build_log_path(log_dir, stage, run_name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove pre-existing handlers so repeated calls don't duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Expose path for downstream code / tests.
    logger.log_path = log_path  # type: ignore[attr-defined]
    return logger


def format_metrics(metrics: Mapping[str, float], prefix: str = "") -> str:
    """Render ``{"L_rec": 0.69, ...}`` as ``"L_rec=0.6900 ..."``."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    body = " ".join(parts)
    return f"{prefix} {body}".strip()
