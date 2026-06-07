"""
config.py
=========

A light-weight, dependency-free configuration system built on top of YAML.

Features
--------
* Nested attribute access:  ``cfg.model.hidden_size``
* Dict-style access:        ``cfg["model"]["hidden_size"]``
* Dotted get/set:           ``cfg.get("model.hidden_size", 768)``
* Config inheritance via a special ``_base_`` key (path(s) to parent YAML(s)).
* Override merging from another mapping or from a list of ``a.b.c=value``
  command line style strings.

The whole point is to make every hyper-parameter in the project overridable
from a single YAML file while keeping the Python side ergonomic.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


# --------------------------------------------------------------------------- #
# Core container                                                              #
# --------------------------------------------------------------------------- #
class Config(MutableMapping):
    """A recursive, attribute-accessible view over a (nested) ``dict``."""

    def __init__(self, data: Mapping | None = None):
        object.__setattr__(self, "_data", {})
        if data:
            for key, value in data.items():
                self[key] = value

    # ---- mapping protocol -------------------------------------------------- #
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        # Recursively wrap nested mappings so attribute access works at depth.
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ---- attribute access -------------------------------------------------- #
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no attribute '{key}'. Available: {list(self._data)}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self._data[key]

    # ---- dotted helpers ---------------------------------------------------- #
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """``cfg.get("a.b.c", default)`` — never raises for a missing path."""
        node: Any = self
        for part in dotted_key.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            elif isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """``cfg.set("a.b.c", value)`` — creates intermediate nodes."""
        parts = dotted_key.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    # ---- conversions ------------------------------------------------------- #
    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self.to_dict()!r})"


# --------------------------------------------------------------------------- #
# Merging                                                                     #
# --------------------------------------------------------------------------- #
def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = dict(base.items()) if isinstance(base, Config) else dict(base)
    for key, value in (override.items() if isinstance(override, Config) else override.items()):
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value.to_dict() if isinstance(value, Config) else value
    return merged


def _coerce_scalar(text: str) -> Any:
    """Best-effort conversion of a CLI string to int / float / bool / None."""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data)}")
    return dict(data)


def load_config(
    path: str,
    overrides: Mapping | None = None,
    cli_overrides: Iterable[str] | None = None,
) -> Config:
    """
    Load a YAML config file.

    Parameters
    ----------
    path:
        Path to the YAML file.
    overrides:
        Optional mapping deep-merged *after* the file is loaded.
    cli_overrides:
        Optional iterable of ``dotted.key=value`` strings (highest priority).

    ``_base_`` handling
    -------------------
    If the YAML contains a ``_base_`` key (a path or list of paths, resolved
    relative to the current file) those files are loaded first and the current
    file is merged on top, enabling preset inheritance.
    """
    raw = _load_yaml(path)

    base_spec = raw.pop("_base_", None)
    merged: dict = {}
    if base_spec is not None:
        base_paths = [base_spec] if isinstance(base_spec, str) else list(base_spec)
        for base_path in base_paths:
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_path)
            parent = load_config(base_path).to_dict()
            merged = deep_merge(merged, parent)
    merged = deep_merge(merged, raw)

    if overrides:
        merged = deep_merge(merged, overrides)

    cfg = Config(merged)

    if cli_overrides:
        for item in cli_overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override '{item}', expected key=value")
            key, value = item.split("=", 1)
            cfg.set(key.strip(), _coerce_scalar(value.strip()))

    return cfg


def save_config(cfg: Config | Mapping, path: str) -> None:
    """Serialise a config (or plain dict) to YAML."""
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
