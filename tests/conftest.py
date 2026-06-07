"""Shared pytest fixtures / path helpers."""

import os
import sys

import pytest

# Make the `newsrec` package importable when running `pytest` from anywhere.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)  # rec/
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TRAIN_DIR = os.path.join(REPO_ROOT, "dataset", "train")
DEV_DIR = os.path.join(REPO_ROOT, "dataset", "dev")


@pytest.fixture(scope="session")
def train_dir() -> str:
    return TRAIN_DIR


@pytest.fixture(scope="session")
def dev_dir() -> str:
    return DEV_DIR


@pytest.fixture
def tiny_news():
    """A handful of synthetic NewsItems for fast, offline tests."""
    from newsrec.data.mind_parser import NewsItem

    return {
        "N1": NewsItem("N1", "sports", "soccer", "Title one", "Abstract one."),
        "N2": NewsItem("N2", "news", "newsworld", "Title two", "Abstract two."),
        "N3": NewsItem("N3", "sports", "nba", "Title three", ""),
        "N4": NewsItem("N4", "finance", "markets", "Title four", "Abstract four."),
    }
