"""
popularity.py
=============

Global item popularity ``p(i)`` for the PAAC popularity-bias machinery.

``p(i)`` is defined (per the PAAC framework) as the click frequency of news
item ``i`` in the training interaction logs.  We count a "click" whenever an
item appears as a *clicked* candidate in an impression; optionally the user
*history* (which is, by construction, a list of previously clicked articles)
can also be counted.

The :class:`ItemPopularity` object then offers everything the downstream
losses need:

* ``count(nid)`` / ``prob(nid)``           — raw count & normalised frequency
* ``top_percent_split(items, x)``          — batch-level top-x% popular split
                                             (used by ``L_cl``)
* ``user_pop_unpop_split(history)``        — per-user popular / unpopular split
                                             (used by ``L_sa``)
* ``long_tail_summary()``                  — head/tail mass + Gini for analysis
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


class ItemPopularity:
    """Holds per-item click counts and derived popularity statistics."""

    def __init__(self, counts: Dict[str, int]):
        self.counts: Dict[str, int] = dict(counts)
        self._total = sum(self.counts.values())

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_impressions(
        cls,
        impressions: Iterable,
        include_history: bool = True,
        include_clicked_candidates: bool = True,
    ) -> "ItemPopularity":
        """Build popularity counts from parsed :class:`Impression` objects."""
        counter: Counter = Counter()
        for imp in impressions:
            if include_history:
                counter.update(imp.history)
            if include_clicked_candidates:
                counter.update(imp.clicked)
        return cls(dict(counter))

    @classmethod
    def from_csv(cls, path: str, id_col: str = "article_id", count_col: str = "clicks") -> "ItemPopularity":
        """Fallback loader for the bundled ``popularity_data.csv``."""
        counts: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                counts[row[id_col]] = int(row[count_col])
        return cls(counts)

    # ------------------------------------------------------------------ #
    # Lookups                                                            #
    # ------------------------------------------------------------------ #
    @property
    def total(self) -> int:
        return self._total

    def count(self, nid: str) -> int:
        return self.counts.get(nid, 0)

    def prob(self, nid: str) -> float:
        """Normalised click frequency ``p(i) = count(i) / sum_j count(j)``."""
        if self._total == 0:
            return 0.0
        return self.counts.get(nid, 0) / self._total

    def counts_for(self, items: Sequence[str]) -> List[int]:
        return [self.count(nid) for nid in items]

    # ------------------------------------------------------------------ #
    # Grouping helpers                                                   #
    # ------------------------------------------------------------------ #
    def top_percent_split(
        self, items: Sequence[str], x: float = 50.0
    ) -> Tuple[List[int], List[int]]:
        """
        Batch-level split used by ``L_cl``.

        Returns ``(popular_idx, unpopular_idx)`` — *positional indices* into
        ``items`` — where the top ``x``% items by global popularity are
        "popular" and the remainder are "unpopular".  Ties are broken by the
        original order so the split is deterministic.
        """
        if not items:
            return [], []
        n = len(items)
        k = int(round(n * x / 100.0))
        k = max(0, min(n, k))
        # Sort positions by popularity (desc), stable on original index.
        order = sorted(range(n), key=lambda i: (-self.count(items[i]), i))
        popular = sorted(order[:k])
        unpopular = sorted(order[k:])
        return popular, unpopular

    def user_pop_unpop_split(
        self, history: Sequence[str], ratio: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """
        Per-user split used by ``L_sa``.

        Splits a user's interacted items into a popular group and an unpopular
        group such that every popular item has strictly-greater-or-equal
        global popularity than every unpopular item (the PAAC ordering
        constraint).  ``ratio`` controls the popular fraction.
        """
        if not history:
            return [], []
        n = len(history)
        k = int(round(n * ratio))
        k = max(0, min(n, k))
        ordered = sorted(history, key=lambda nid: (-self.count(nid), nid))
        return ordered[:k], ordered[k:]

    # ------------------------------------------------------------------ #
    # Analysis                                                           #
    # ------------------------------------------------------------------ #
    def most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        return Counter(self.counts).most_common(n)

    def long_tail_summary(self, head_frac: float = 0.2) -> Dict[str, float]:
        """Head/tail click-mass and Gini coefficient over the click counts."""
        if self._total == 0:
            return {"num_items": 0, "head_mass": 0.0, "tail_mass": 0.0, "gini": 0.0}
        sorted_counts = sorted(self.counts.values(), reverse=True)
        n = len(sorted_counts)
        head_n = max(1, int(round(n * head_frac)))
        head_mass = sum(sorted_counts[:head_n]) / self._total
        return {
            "num_items": n,
            "head_frac": head_frac,
            "head_mass": round(head_mass, 4),
            "tail_mass": round(1.0 - head_mass, 4),
            "gini": round(_gini(sorted_counts), 4),
        }


def _gini(values: Sequence[float]) -> float:
    """Gini coefficient of a list of non-negative values."""
    if not values:
        return 0.0
    ascending = sorted(values)
    n = len(ascending)
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(ascending, start=1):
        cum += v
        weighted += i * v
    if cum == 0:
        return 0.0
    return (2.0 * weighted) / (n * cum) - (n + 1.0) / n


def build_popularity(
    data_dir: str | None = None,
    impressions: Iterable | None = None,
    csv_fallback: bool = True,
    **kwargs,
) -> ItemPopularity:
    """
    Convenience builder.

    Prefers computing from ``impressions``.  If only a ``data_dir`` is given,
    loads the behaviours there.  As a last resort (``csv_fallback``) loads the
    bundled ``popularity_data.csv``.
    """
    if impressions is not None:
        return ItemPopularity.from_impressions(impressions, **kwargs)
    if data_dir is not None:
        behaviors = os.path.join(data_dir, "behaviors.tsv")
        if os.path.exists(behaviors):
            from newsrec.data.mind_parser import parse_behaviors

            return ItemPopularity.from_impressions(parse_behaviors(behaviors), **kwargs)
        csv_path = os.path.join(data_dir, "popularity_data.csv")
        if csv_fallback and os.path.exists(csv_path):
            return ItemPopularity.from_csv(csv_path)
    raise ValueError("Provide `impressions` or a `data_dir` containing behaviors.tsv")
