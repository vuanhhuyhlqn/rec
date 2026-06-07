"""Tests for newsrec.data.popularity."""

import os

import pytest

from newsrec.data.mind_parser import Impression, load_mind_split
from newsrec.data.popularity import ItemPopularity, build_popularity


def _make_impressions():
    # N1 very popular, N2 medium, N3 rare.
    return [
        Impression("1", "U1", "t", ["N1", "N2"], [("N1", 1), ("N3", 0)]),
        Impression("2", "U2", "t", ["N1"], [("N2", 1), ("N4", 0)]),
        Impression("3", "U3", "t", ["N1", "N2"], [("N1", 1), ("N5", 0)]),
    ]


def test_counts_and_prob():
    pop = ItemPopularity.from_impressions(_make_impressions())
    # history: N1 x3, N2 x2 ; clicked: N1 x2, N2 x1
    assert pop.count("N1") == 5
    assert pop.count("N2") == 3
    assert pop.count("N3") == 0  # only appeared as non-clicked candidate
    assert pop.total == 8
    assert pop.prob("N1") == pytest.approx(5 / 8)
    assert pop.prob("missing") == 0.0


def test_counts_clicked_only():
    pop = ItemPopularity.from_impressions(_make_impressions(), include_history=False)
    assert pop.count("N1") == 2
    assert pop.count("N2") == 1
    assert pop.total == 3


def test_top_percent_split():
    pop = ItemPopularity.from_impressions(_make_impressions())
    items = ["N3", "N1", "N2", "N4"]  # counts: 0, 6, 3, 0
    popular, unpopular = pop.top_percent_split(items, x=50.0)
    # top 50% (2 of 4) most popular = N1(idx1), N2(idx2)
    assert popular == [1, 2]
    assert unpopular == [0, 3]
    # disjoint and complete
    assert sorted(popular + unpopular) == [0, 1, 2, 3]


def test_top_percent_edge_cases():
    pop = ItemPopularity.from_impressions(_make_impressions())
    assert pop.top_percent_split([], x=50.0) == ([], [])
    pop_idx, unpop_idx = pop.top_percent_split(["N1"], x=50.0)
    assert sorted(pop_idx + unpop_idx) == [0]


def test_user_pop_unpop_split_ordering():
    pop = ItemPopularity.from_impressions(_make_impressions())
    history = ["N3", "N1", "N2"]  # counts 0, 6, 3
    popular, unpopular = pop.user_pop_unpop_split(history, ratio=0.5)
    # ordering constraint: min popular count >= max unpopular count
    assert min(pop.count(i) for i in popular) >= max(pop.count(i) for i in unpopular)


def test_long_tail_summary():
    pop = ItemPopularity.from_impressions(_make_impressions())
    summary = pop.long_tail_summary()
    assert summary["num_items"] >= 1
    assert 0.0 <= summary["gini"] <= 1.0
    assert summary["head_mass"] + summary["tail_mass"] == pytest.approx(1.0, abs=1e-6)


def test_csv_fallback(tmp_path):
    csv_path = tmp_path / "popularity_data.csv"
    csv_path.write_text("rank,article_id,clicks\n1,N1,100\n2,N2,40\n")
    pop = ItemPopularity.from_csv(str(csv_path))
    assert pop.count("N1") == 100
    assert pop.count("N2") == 40


def test_build_popularity_real(train_dir):
    if not os.path.exists(os.path.join(train_dir, "behaviors.tsv")):
        pytest.skip("MINDsmall_train not present")
    data = load_mind_split(train_dir)
    pop = build_popularity(impressions=data.impressions)
    assert pop.total > 0
    top = pop.most_common(10)
    assert len(top) == 10
    # head mass should dominate for a long-tail dataset
    summary = pop.long_tail_summary()
    assert summary["head_mass"] > 0.4
