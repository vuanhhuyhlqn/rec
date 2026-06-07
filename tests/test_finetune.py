"""Tests for Task 8: finetune dataset, BPR loss, and the train loop."""

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.finetune_dataset import FinetuneTripletDataset
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.losses.paac_losses import bpr_loss
from newsrec.models.rec_model import build_rec_model
from newsrec.training.finetuner import Finetuner


class _Impr:
    def __init__(self, history, candidates):
        self.history = history
        self.candidates = candidates

    @property
    def clicked(self):
        return [n for n, l in self.candidates if l == 1]

    @property
    def non_clicked(self):
        return [n for n, l in self.candidates if l == 0]


def _fake_token_table(num_news=12, max_len=8):
    tokens = {
        f"N{i}": {
            "input_ids": [1 + (i % 5)] * max_len,
            "attention_mask": [1] * max_len,
        }
        for i in range(num_news)
    }
    return NewsTokenTable(tokens, max_len)


def _impressions():
    return [
        _Impr(["N0", "N1", "N2"], [("N3", 1), ("N4", 0), ("N5", 0)]),
        _Impr(["N6"], [("N7", 1), ("N8", 0)]),
        _Impr(["N9", "N10"], [("N11", 0), ("N2", 0)]),  # no click -> skipped
    ]


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
def test_dataset_builds_valid_triplets():
    table = _fake_token_table()
    ds = FinetuneTripletDataset(_impressions(), table, max_history=5)
    # Only the first two impressions yield triplets (third has no click).
    assert len(ds) == 2
    # each entry stores (history, pos, non_clicked_pool); the sampled negative
    # must be a non-clicked item of the same impression.
    for i, (history, pos, non_clicked) in enumerate(ds.triplets):
        assert pos in {"N3", "N7"}
        assert set(non_clicked) <= {"N4", "N5", "N8"}
        assert ds._sample_neg(non_clicked, i) in {"N4", "N5", "N8"}


def test_negative_resampling_varies_by_epoch():
    table = _fake_token_table()
    ds = FinetuneTripletDataset(_impressions(), table, max_history=5,
                                negatives_per_pos=1, resample_negatives=True)
    # Deterministic within an epoch, reproducible across calls.
    ds.set_epoch(0)
    e0 = [ds._sample_neg(nc, i) for i, (_, _, nc) in enumerate(ds.triplets)]
    assert e0 == [ds._sample_neg(nc, i) for i, (_, _, nc) in enumerate(ds.triplets)]
    # Across many epochs at least one sampled negative changes (when a pool of
    # >1 candidates exists).
    seqs = []
    for ep in range(8):
        ds.set_epoch(ep)
        seqs.append(tuple(ds._sample_neg(nc, i) for i, (_, _, nc) in enumerate(ds.triplets)))
    assert len(set(seqs)) > 1, "negatives should be reshuffled across epochs"

    # With resampling off, the negative is fixed across epochs.
    ds_fixed = FinetuneTripletDataset(_impressions(), table, max_history=5,
                                      resample_negatives=False)
    ds_fixed.set_epoch(0)
    a = [ds_fixed._sample_neg(nc, i) for i, (_, _, nc) in enumerate(ds_fixed.triplets)]
    ds_fixed.set_epoch(5)
    b = [ds_fixed._sample_neg(nc, i) for i, (_, _, nc) in enumerate(ds_fixed.triplets)]
    assert a == b


def test_dataset_item_shapes():
    table = _fake_token_table(max_len=8)
    ds = FinetuneTripletDataset(_impressions(), table, max_history=5)
    sample = ds[0]
    assert sample["history_input_ids"].shape == (5, 8)
    assert sample["history_attention_mask"].shape == (5, 8)
    assert sample["history_mask"].shape == (5,)
    assert sample["pos_input_ids"].shape == (8,)
    assert sample["neg_input_ids"].shape == (8,)
    # history mask marks the real (<=3) history positions for impression 0
    assert sample["history_mask"].sum() <= 5


def test_collate_batches():
    table = _fake_token_table()
    ds = FinetuneTripletDataset(_impressions(), table, max_history=5)
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)
    batch = next(iter(loader))
    assert batch["history_input_ids"].shape[0] == 2
    assert batch["pos_input_ids"].shape == (2, 8)


# --------------------------------------------------------------------------- #
# BPR loss                                                                    #
# --------------------------------------------------------------------------- #
def test_bpr_loss_positive_and_decreases():
    pos_low = torch.tensor([0.1, 0.2])
    neg_high = torch.tensor([0.9, 0.8])
    pos_high = torch.tensor([0.9, 0.8])
    neg_low = torch.tensor([0.1, 0.2])
    loss_bad = bpr_loss(pos_low, neg_high)
    loss_good = bpr_loss(pos_high, neg_low)
    assert loss_bad.item() > 0 and loss_good.item() > 0
    assert loss_good.item() < loss_bad.item()


# --------------------------------------------------------------------------- #
# Train loop                                                                  #
# --------------------------------------------------------------------------- #
def _tiny_model():
    cfg = {
        "plm": {"pretrained": False, "use_lora": True, "lora_r": 4,
                "small_config": dict(hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=4, intermediate_size=64,
                                     max_position_embeddings=16, vocab_size=50)},
        "model_dim": 32,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "max_title_len": 8,
        "max_history_len": 5,
    }
    return build_rec_model(cfg)


def test_finetuner_train_step_finite_and_learns():
    torch.manual_seed(0)
    model = _tiny_model()
    table = _fake_token_table(max_len=8)
    ds = FinetuneTripletDataset(_impressions(), table, max_history=5, negatives_per_pos=4)
    loader = DataLoader(ds, batch_size=4, collate_fn=stack_collate, shuffle=True)
    tuner = Finetuner(model, config={"lr": 1e-2, "lambda3": 0.0, "grad_clip": 1.0})

    batch = next(iter(loader))
    first = tuner.train_step(batch)
    assert first["total"] == first["total"]  # not NaN
    losses = [first["total"]]
    for _ in range(20):
        losses.append(tuner.train_step(batch)["total"])
    # Loss should trend downward on a fixed batch.
    assert losses[-1] < losses[0]
