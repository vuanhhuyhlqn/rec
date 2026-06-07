"""
mind_parser.py
==============

Parsers for the MIND (Microsoft News Dataset) raw TSV files.

File formats (verified against ``MINDsmall_train``)::

    news.tsv:
        nid \\t category \\t subcategory \\t title \\t abstract \\t url
            \\t title_entities \\t abstract_entities

    behaviors.tsv:
        impr_id \\t user_id \\t time \\t history \\t impressions
        history     := space-separated news ids (may be empty)
        impressions := space-separated "NID-{0|1}" tokens
                       (1 = clicked, 0 = not clicked)

The parser is intentionally tokenizer-free so it can be exercised in unit
tests without network access.  Text tokenisation lives in
:mod:`newsrec.data.vocab` / model code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class NewsItem:
    """A single news article."""

    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str

    @property
    def text(self) -> str:
        """Concatenated title + abstract used as model input."""
        if self.abstract:
            return f"{self.title} {self.abstract}".strip()
        return self.title.strip()


@dataclass
class Impression:
    """A single impression / behaviour log row."""

    impression_id: str
    user_id: str
    time: str
    history: List[str]
    candidates: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def clicked(self) -> List[str]:
        return [nid for nid, label in self.candidates if label == 1]

    @property
    def non_clicked(self) -> List[str]:
        return [nid for nid, label in self.candidates if label == 0]


def parse_news(path: str) -> Dict[str, NewsItem]:
    """Parse ``news.tsv`` into ``{news_id: NewsItem}``."""
    news: Dict[str, NewsItem] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # Defensive: a few rows may miss trailing fields.
            if len(parts) < 5:
                continue
            nid, category, subcategory, title, abstract = parts[:5]
            news[nid] = NewsItem(
                news_id=nid,
                category=category,
                subcategory=subcategory,
                title=title,
                abstract=abstract,
            )
    return news


def parse_behaviors(path: str) -> List[Impression]:
    """Parse ``behaviors.tsv`` into a list of :class:`Impression`."""
    impressions: List[Impression] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                # Some rows can have an empty history but should still have 5 cols.
                # Pad missing trailing columns with empty strings.
                parts = parts + [""] * (5 - len(parts))
            impr_id, user_id, time, history_str, impr_str = parts[:5]

            history = history_str.split() if history_str.strip() else []

            candidates: List[Tuple[str, int]] = []
            for token in impr_str.split():
                if "-" not in token:
                    continue
                nid, label = token.rsplit("-", 1)
                candidates.append((nid, int(label)))

            impressions.append(
                Impression(
                    impression_id=impr_id,
                    user_id=user_id,
                    time=time,
                    history=history,
                    candidates=candidates,
                )
            )
    return impressions


@dataclass
class MindData:
    """Container bundling parsed news + impressions for one split."""

    news: Dict[str, NewsItem]
    impressions: List[Impression]

    # ---- convenience accessors -------------------------------------------- #
    @property
    def categories(self) -> List[str]:
        return sorted({item.category for item in self.news.values()})

    @property
    def subcategories(self) -> List[str]:
        return sorted({item.subcategory for item in self.news.values()})

    @property
    def users(self) -> List[str]:
        return sorted({imp.user_id for imp in self.impressions})

    def avg_history_length(self) -> float:
        if not self.impressions:
            return 0.0
        total = sum(len(imp.history) for imp in self.impressions)
        return total / len(self.impressions)

    def stats(self) -> Dict[str, float]:
        n_candidates = sum(len(imp.candidates) for imp in self.impressions)
        n_clicks = sum(len(imp.clicked) for imp in self.impressions)
        return {
            "num_news": len(self.news),
            "num_impressions": len(self.impressions),
            "num_users": len(self.users),
            "num_categories": len(self.categories),
            "num_subcategories": len(self.subcategories),
            "avg_history_length": round(self.avg_history_length(), 3),
            "num_candidates": n_candidates,
            "num_clicks": n_clicks,
        }


def load_mind_split(data_dir: str) -> MindData:
    """Load ``news.tsv`` + ``behaviors.tsv`` from a MIND split directory."""
    news_path = os.path.join(data_dir, "news.tsv")
    behaviors_path = os.path.join(data_dir, "behaviors.tsv")
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"news.tsv not found in {data_dir}")
    if not os.path.exists(behaviors_path):
        raise FileNotFoundError(f"behaviors.tsv not found in {data_dir}")
    return MindData(
        news=parse_news(news_path),
        impressions=parse_behaviors(behaviors_path),
    )
