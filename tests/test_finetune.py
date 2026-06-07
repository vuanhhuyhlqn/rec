"""Tests for Task 8: finetune dataset, BPR loss, LoRA scheduler, train loop."""

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.finetune_dataset import FinetuneTripletDataset
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.losses.paac_losses import bpr_loss
from newsrec.models.plm_encoder import PLMEncoder
from newsrec.models.rec_model import build_rec_model
from newsrec.training.finetuner import Finetuner
from newsrec.training.lora_schedule import LoRAUnfreezeScheduler


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
    # negatives must be non-clicked in the same impression
    for history, pos, neg in ds.triplets:
        assert pos in {"N3", "N7"}
        assert neg in {"N4", "N5", "N8"}


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
# LoRA scheduler                                                              #
# --------------------------------------------------------------------------- #
def test_lora_scheduler_unfreezes_expected_layers():
    plm = PLMEncoder(
        pretrained=False, use_lora=True, lora_r=4,
        small_config=dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
                          intermediate_size=64, max_position_embeddings=32, vocab_size=100),
    )
    sched = LoRAUnfreezeScheduler(plm, schedule=[[0, 0], [2, 2], [4, 6]])
    changed0, n0 = sched.step(0)
    assert changed0 and n0 == 0 and plm.frozen_layers == 6
    changed1, n1 = sched.step(1)
    assert not changed1 and n1 == 0  # still LoRA-only
    changed2, n2 = sched.step(2)
    assert changed2 and n2 == 2 and plm.frozen_layers == 4
    changed4, n4 = sched.step(5)
    assert changed4 and n4 == 6 and plm.frozen_layers == 0


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
