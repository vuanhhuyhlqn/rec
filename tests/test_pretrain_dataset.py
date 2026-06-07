"""Tests for Task 10: pretrain dataset generation."""

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.data.pretrain_dataset import PretrainDataset, build_user_sequences


class _Impr:
    def __init__(self, user_id, history, candidates=()):
        self.user_id = user_id
        self.history = list(history)
        self.candidates = list(candidates)


def _table(num=20, max_len=8):
    tokens = {f"N{i}": {"input_ids": [1 + i % 4] * max_len,
                        "attention_mask": [1] * max_len} for i in range(num)}
    return NewsTokenTable(tokens, max_len)


def _category_ids(num=20):
    # categories 2..6 (0=PAD,1=UNK reserved)
    return {f"N{i}": 2 + (i % 5) for i in range(num)}


def test_build_user_sequences_keeps_longest():
    imprs = [
        _Impr("U1", ["N0", "N1"]),
        _Impr("U1", ["N0", "N1", "N2", "N3"]),  # longest for U1
        _Impr("U2", ["N4"]),  # too short -> dropped (min_len=3)
    ]
    table = _table()
    seqs = build_user_sequences(imprs, min_len=3, in_table=table.has)
    users = {u: s for u, s in seqs}
    assert "U1" in users and users["U1"] == ["N0", "N1", "N2", "N3"]
    assert "U2" not in users  # filtered by min_len


def test_pretrain_item_shapes():
    table = _table(max_len=8)
    seqs = [("U1", [f"N{i}" for i in range(6)])]
    ds = PretrainDataset(seqs, table, _category_ids(), max_seq_len=10, mask_prob=0.15)
    item = ds[0]
    S, L = 10, 8
    assert item["input_ids"].shape == (S, L)
    assert item["attention_mask"].shape == (S, L)
    for key in ("seq_mask", "mip_mask", "category", "segment_mask",
                "context_mask", "bsm_a_mask", "bsm_b_mask"):
        assert item[key].shape == (S,)
    # 6 valid positions
    assert item["seq_mask"].sum().item() == 6


def test_at_least_one_masked_and_within_valid():
    table = _table()
    seqs = [("U1", [f"N{i}" for i in range(5)])]
    ds = PretrainDataset(seqs, table, _category_ids(), max_seq_len=10, mask_prob=0.0)
    item = ds[0]
    # mask_prob=0 still forces exactly one masked position (the last valid one)
    assert item["mip_mask"].sum().item() >= 1
    # masked positions must be valid history positions
    assert torch.all((item["mip_mask"] * (1 - item["seq_mask"])) == 0)


def test_segment_is_contiguous_and_removed_from_context():
    table = _table()
    seqs = [("U1", [f"N{i}" for i in range(8)])]
    ds = PretrainDataset(seqs, table, _category_ids(), max_seq_len=10)
    item = ds[0]
    seg = item["segment_mask"]
    # contiguous: indices of 1s form a continuous run
    idx = torch.nonzero(seg).flatten().tolist()
    assert idx == list(range(idx[0], idx[-1] + 1))
    # context excludes segment positions
    assert torch.all(item["context_mask"] * seg == 0)
    # context + segment (restricted to valid) == seq_mask
    assert torch.allclose(item["context_mask"] + seg * item["seq_mask"], item["seq_mask"])


def test_bsm_non_overlapping_same_sequence():
    table = _table()
    seqs = [("U1", [f"N{i}" for i in range(7)])]
    ds = PretrainDataset(seqs, table, _category_ids(), max_seq_len=10)
    item = ds[0]
    a, b = item["bsm_a_mask"], item["bsm_b_mask"]
    assert torch.all(a * b == 0)  # disjoint
    # together they cover all 7 valid positions
    assert (a + b).sum().item() == 7


def test_category_targets_match():
    table = _table()
    cat = _category_ids()
    seqs = [("U1", ["N0", "N1", "N2"])]
    ds = PretrainDataset(seqs, table, cat, max_seq_len=5)
    item = ds[0]
    assert item["category"][0].item() == cat["N0"]
    assert item["category"][2].item() == cat["N2"]
    # padded positions have category 0
    assert item["category"][3].item() == 0


def test_collate_batches_pretrain():
    table = _table()
    seqs = [("U1", [f"N{i}" for i in range(6)]), ("U2", [f"N{i}" for i in range(3, 9)])]
    ds = PretrainDataset(seqs, table, _category_ids(), max_seq_len=10)
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 10, 8)
    assert batch["mip_mask"].shape == (2, 10)
