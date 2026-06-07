"""
env.py
======

Minimal ``.env`` loading and HuggingFace token resolution (no external
dependency).

* :func:`load_dotenv` parses a simple ``KEY=VALUE`` file into ``os.environ``
  (without overwriting variables already set in the real environment).
* :func:`get_hf_token` returns the HF token, accepting either the
  ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN`` variable name.

Secrets are never logged; only their presence is reported by callers.
"""

from __future__ import annotations

import os
from typing import Optional

HF_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    """
    Load ``KEY=VALUE`` pairs from ``path`` into ``os.environ``.

    Returns True if the file existed and was read.  Existing environment
    variables are preserved unless ``override`` is True.
    """
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
    return True


def get_hf_token() -> Optional[str]:
    """Return the HuggingFace token from any of the accepted env var names."""
    for var in HF_TOKEN_VARS:
        token = os.environ.get(var)
        if token:
            return token
    return None
