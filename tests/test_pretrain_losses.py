"""Tests for Task 11: pre-train losses + config selection + PretrainModule."""

import math

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.data.pretrain_dataset import PretrainDataset
from newsrec.losses.infonce import info_nce_against_table, info_nce_inbatch
from newsrec.losses.pretrain_losses import (
    PRETRAIN_TASKS,
    aap_loss,
    bsm_loss,
    map_loss,
    mip_loss,
    select_enabled_tasks,
    sp_loss,
    task_weights,
)
from newsrec.models.pretrain_model import PretrainModule
from newsrec.models.rec_model import build_rec_model


# --------------------------------------------------------------------------- #
# InfoNCE exactness                                                           #
# --------------------------------------------------------------------------- #
def test_info_nce_inbatch_exact_value():
    # Two orthonormal rows, query == key, tau = 1.
    q = torch.eye(2)
    loss = info_nce_inbatch(q, q, tau=1.0)
    # logits = I; per-row CE = -log(e^1/(e^1+e^0)) = log(1 + e^-1)
    expected = math.log(1 + math.exp(-1))
    assert math.isclose(loss.item(), expected, abs_tol=1e-5)


def test_info_nce_inbatch_small_batch_zero():
    assert info_nce_inbatch(torch.randn(1, 4), torch.randn(1, 4)).item() == 0.0


def test_info_nce_against_table_perfect():
    table = torch.eye(3)
    anchors = torch.eye(3)  # each anchor aligned with its category row
    labels = torch.arange(3)
    loss_aligned = info_nce_against_table(anchors, labels, table, tau=0.1)
    # Wrong labels → higher loss.
    loss_wrong = info_nce_against_table(anchors, torch.tensor([1, 2, 0]), table, tau=0.1)
    assert loss_aligned.item() < loss_wrong.item()


# --------------------------------------------------------------------------- #
# Individual losses: finite & shaped                                          #
# --------------------------------------------------------------------------- #
def test_each_loss_finite():
    D, C = 8, 5
    news = torch.randn(6, D)
    cats = torch.randint(0, C, (6,))
    table = torch.randn(C, D)
    assert torch.isfinite(aap_loss(news, cats, table))
    assert torch.isfinite(mip_loss(torch.randn(4, D), torch.randn(4, D)))
    assert torch.isfinite(map_loss(torch.randn(4, D), torch.randint(0, C, (4,)), table))
    assert torch.isfinite(sp_loss(torch.randn(3, D), torch.randn(3, D)))
    assert torch.isfinite(bsm_loss(torch.randn(3, D), torch.randn(3, D)))


# --------------------------------------------------------------------------- #
# Registry / config selection                                                 #
# --------------------------------------------------------------------------- #
def test_select_enabled_tasks_list_and_mapping():
    assert select_enabled_tasks(["aap", "mip"]) == ["aap", "mip"]
    mapping = {"aap": {"enabled": True, "weight": 2.0},
               "mip": {"enabled": False},
               "bsm": True}
    enabled = select_enabled_tasks(mapping)
    assert "aap" in enabled and "bsm" in enabled and "mip" not in enabled
    w = task_weights(mapping, enabled)
    assert w["aap"] == 2.0 and w["bsm"] == 1.0


def test_select_unknown_task_raises():
    try:
        select_enabled_tasks(["bogus"])
        assert False, "should raise"
    except ValueError:
        pass


def test_all_tasks_registered():
    assert set(PRETRAIN_TASKS) == {"aap", "mip", "map", "sp", "bsm"}


# --------------------------------------------------------------------------- #
# PretrainModule integration                                                  #
# --------------------------------------------------------------------------- #
def _setup_module(enabled):
    max_len = 8
    tokens = {f"N{i}": {"input_ids": [1 + i % 4] * max_len,
                        "attention_mask": [1] * max_len} for i in range(20)}
    table = NewsTokenTable(tokens, max_len)
    cat_ids = {f"N{i}": 2 + (i % 4) for i in range(20)}
    seqs = [("U1", [f"N{i}" for i in range(6)]),
            ("U2", [f"N{i}" for i in range(4, 12)])]
    ds = PretrainDataset(seqs, table, cat_ids, max_seq_len=10, mask_prob=0.3)
    cfg = {
        "plm": {"pretrained": False, "use_lora": True, "lora_r": 4,
                "small_config": dict(hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=4, intermediate_size=64,
                                     max_position_embeddings=16, vocab_size=50)},
        "model_dim": 32,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "max_title_len": 8, "max_history_len": 10,
    }
    model = build_rec_model(cfg)
    module = PretrainModule(model, num_categories=8, enabled_tasks=enabled, tau=0.1)
    return module, ds


def test_pretrain_module_all_tasks():
    torch.manual_seed(0)
    module, ds = _setup_module(["aap", "mip", "map", "sp", "bsm"])
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)
    batch = next(iter(loader))
    losses = module.compute_losses(batch)
    for key in ("aap", "mip", "map", "sp", "bsm", "total"):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert losses["total"].item() > 0


def test_pretrain_module_subset_only():
    torch.manual_seed(0)
    module, ds = _setup_module(["aap", "mip", "bsm"])
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)
    batch = next(iter(loader))
    losses = module.compute_losses(batch)
    assert set(losses.keys()) == {"aap", "mip", "bsm", "total"}
    assert "map" not in losses and "sp" not in losses


def test_pretrain_gradients_reach_encoder():
    torch.manual_seed(0)
    module, ds = _setup_module(["aap", "mip", "map", "sp", "bsm"])
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)
    batch = next(iter(loader))
    losses = module.compute_losses(batch)
    losses["total"].backward()
    # category table + mask token receive gradient
    assert module.category_embeddings.weight.grad is not None
    assert module.mask_token.grad is not None
    # at least some LoRA params in the PLM receive gradient
    lora_grads = [p.grad for n, p in module.model.named_parameters()
                  if p.requires_grad and "lora_B" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)
