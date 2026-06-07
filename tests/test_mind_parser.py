"""Tests for newsrec.data.mind_parser and newsrec.data.vocab."""

import os

import pytest

from newsrec.data.mind_parser import (
    Impression,
    NewsItem,
    load_mind_split,
    parse_behaviors,
    parse_news,
)
from newsrec.data.vocab import (
    MASK_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocab,
    build_category_vocab,
    build_news_vocab,
)


# --------------------------------------------------------------------------- #
# Synthetic / offline tests                                                   #
# --------------------------------------------------------------------------- #
def test_newsitem_text_concatenation():
    item = NewsItem("N1", "sports", "nba", "Hello", "World.")
    assert item.text == "Hello World."
    item2 = NewsItem("N2", "sports", "nba", "Only title", "")
    assert item2.text == "Only title"


def test_parse_behaviors_tmpfile(tmp_path):
    path = tmp_path / "behaviors.tsv"
    path.write_text(
        "1\tU1\t11/11/2019\tN1 N2 N3\tN4-1 N5-0 N6-0\n"
        "2\tU2\t11/12/2019\t\tN7-0 N8-1\n"  # empty history
    )
    imps = parse_behaviors(str(path))
    assert len(imps) == 2
    assert imps[0].history == ["N1", "N2", "N3"]
    assert imps[0].candidates == [("N4", 1), ("N5", 0), ("N6", 0)]
    assert imps[0].clicked == ["N4"]
    assert imps[0].non_clicked == ["N5", "N6"]
    assert imps[1].history == []  # empty history handled


def test_parse_news_tmpfile(tmp_path):
    path = tmp_path / "news.tsv"
    path.write_text(
        "N1\tsports\tnba\tTitle1\tAbs1\thttp://u\t[]\t[]\n"
        "N2\tnews\tworld\tTitle2\tAbs2\thttp://u\t[]\t[]\n"
    )
    news = parse_news(str(path))
    assert set(news) == {"N1", "N2"}
    assert news["N1"].category == "sports"
    assert news["N2"].title == "Title2"


def test_vocab_special_tokens_and_unk():
    vocab = Vocab("news", special_tokens=(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN))
    vocab.build(["N1", "N2", "N1"])  # duplicate ignored
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.mask_id == 2
    assert len(vocab) == 5  # 3 special + N1 + N2
    assert vocab.index("N1") == 3
    assert vocab.index("UNKNOWN_NEWS") == vocab.unk_id  # falls back to UNK
    assert vocab.token(3) == "N1"


def test_vocab_save_load(tmp_path):
    vocab = build_news_vocab({"N1": None, "N2": None})
    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    reloaded = Vocab.load(str(path))
    assert reloaded.token2id == vocab.token2id
    assert reloaded.id2token == vocab.id2token


def test_build_vocabs(tiny_news):
    news_vocab = build_news_vocab(tiny_news)
    cat_vocab = build_category_vocab(tiny_news)
    # 3 specials + 4 news
    assert len(news_vocab) == 3 + 4
    # 2 specials + {sports, news, finance}
    assert len(cat_vocab) == 2 + 3
    assert "sports" in cat_vocab


# --------------------------------------------------------------------------- #
# Real-data tests (skipped if data is absent)                                 #
# --------------------------------------------------------------------------- #
def _has_data(train_dir):
    return os.path.exists(os.path.join(train_dir, "news.tsv"))


def test_real_mind_train_loads(train_dir):
    if not _has_data(train_dir):
        pytest.skip("MINDsmall_train not present")
    data = load_mind_split(train_dir)
    stats = data.stats()
    assert stats["num_news"] > 1000
    assert stats["num_impressions"] > 1000
    # MIND has ~17 top-level categories.
    assert 10 <= stats["num_categories"] <= 25
    assert stats["avg_history_length"] > 0


def test_real_history_candidates_resolve(train_dir):
    if not _has_data(train_dir):
        pytest.skip("MINDsmall_train not present")
    data = load_mind_split(train_dir)
    news_vocab = build_news_vocab(data.news)
    # Every candidate id in the first 200 impressions should be encodable
    # (either known or mapped to UNK without raising).
    for imp in data.impressions[:200]:
        for nid, label in imp.candidates:
            assert label in (0, 1)
            assert isinstance(news_vocab.index(nid), int)
