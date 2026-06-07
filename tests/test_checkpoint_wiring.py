"""Tests for Task 12: checkpointing, async HF uploader, pretrain->finetune wiring."""

import os

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.finetune_dataset import FinetuneTripletDataset
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.data.popularity import ItemPopularity
from newsrec.data.pretrain_dataset import PretrainDataset
from newsrec.models.pretrain_model import PretrainModule
from newsrec.models.rec_model import build_rec_model
from newsrec.training.checkpoint import (
    CheckpointManager,
    load_backbone_weights,
    save_checkpoint,
)
from newsrec.training.finetuner import Finetuner
from newsrec.training.hub_uploader import HubUploader
from newsrec.training.pretrainer import Pretrainer


SMALL_PLM = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                 intermediate_size=64, max_position_embeddings=16, vocab_size=50)


def _model():
    cfg = {
        "plm": {"pretrained": False, "use_lora": True, "lora_r": 4,
                "small_config": SMALL_PLM},
        "model_dim": 32,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "max_title_len": 8, "max_history_len": 10,
    }
    return build_rec_model(cfg)


def _table(num=20, max_len=8):
    tokens = {f"N{i}": {"input_ids": [1 + i % 4] * max_len,
                        "attention_mask": [1] * max_len} for i in range(num)}
    return NewsTokenTable(tokens, max_len)


# --------------------------------------------------------------------------- #
# Checkpoint round-trip                                                       #
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    model = _model()
    out = save_checkpoint(model, str(tmp_path / "ckpt"))
    assert os.path.exists(os.path.join(out, "model.pt"))

    model2 = _model()
    # weights differ initially
    p1 = dict(model.named_parameters())
    p2 = dict(model2.named_parameters())
    key = "user_encoder.pooler.fc2.weight"
    assert not torch.allclose(p1[key], p2[key])

    load_backbone_weights(model2, out, strict=False)
    p2 = dict(model2.named_parameters())
    assert torch.allclose(p1[key], p2[key], atol=1e-6)


# --------------------------------------------------------------------------- #
# HF uploader (stubbed, no network)                                           #
# --------------------------------------------------------------------------- #
class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploads = []

    def create_repo(self, repo_id, private=False, exist_ok=True):
        self.created.append((repo_id, private))

    def upload_folder(self, repo_id, folder_path, path_in_repo):
        self.uploads.append((repo_id, folder_path, path_in_repo))


def test_hub_uploader_async_invocation(tmp_path):
    api = _FakeApi()
    up = HubUploader(repo_id="user/newsrec", token="fake-token", enabled=True, api=api)
    assert up.enabled
    folder = str(tmp_path / "epoch0")
    os.makedirs(folder, exist_ok=True)
    assert up.enqueue(folder, "epoch0") is True
    up.close(wait=True)  # flush background thread
    assert api.uploads == [("user/newsrec", folder, "epoch0")]
    assert api.created and api.created[0][0] == "user/newsrec"


def test_hub_uploader_graceful_fallback_no_token():
    # enabled requested but no token + no api -> disabled, enqueue is a no-op.
    up = HubUploader(repo_id="user/newsrec", token=None, enabled=True)
    assert up.enabled is False
    assert up.enqueue("/tmp/whatever", "epoch0") is False
    up.close()  # should not raise


def test_checkpoint_manager_best_tracking(tmp_path):
    api = _FakeApi()
    up = HubUploader(repo_id="user/newsrec", token="fake", enabled=True, api=api)
    mgr = CheckpointManager(str(tmp_path / "ck"), config={"a": 1}, uploader=up)
    model = _model()
    mgr.save(model, tag="epoch0", metric=0.5, higher_is_better=True)
    mgr.save(model, tag="epoch1", metric=0.7, higher_is_better=True)  # new best
    mgr.save(model, tag="epoch2", metric=0.6, higher_is_better=True)  # not best
    mgr.close()
    assert mgr.best_metric == 0.7
    assert os.path.exists(tmp_path / "ck" / "best" / "model.pt")
    # uploads: epoch0, best(0.5), epoch1, best(0.7), epoch2  -> contains a 'best'
    paths = [u[2] for u in api.uploads]
    assert "best" in paths and "epoch1" in paths


# --------------------------------------------------------------------------- #
# Pretrain -> finetune wiring                                                 #
# --------------------------------------------------------------------------- #
def test_pretrain_then_finetune_loads_backbone(tmp_path):
    torch.manual_seed(0)
    table = _table()
    cat_ids = {f"N{i}": 2 + (i % 4) for i in range(20)}
    seqs = [("U1", [f"N{i}" for i in range(6)]), ("U2", [f"N{i}" for i in range(4, 12)])]
    ds = PretrainDataset(seqs, table, cat_ids, max_seq_len=10, mask_prob=0.3)
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)

    model = _model()
    module = PretrainModule(model, num_categories=8, enabled_tasks=["aap", "mip", "bsm"])
    mgr = CheckpointManager(str(tmp_path / "pre"))
    trainer = Pretrainer(module, config={"lr": 1e-2, "epochs": 1, "save_every": 1},
                         checkpoint_manager=mgr)
    trainer.train(loader)
    ckpt_dir = os.path.join(str(tmp_path / "pre"), "epoch0")
    assert os.path.exists(os.path.join(ckpt_dir, "model.pt"))

    # New finetune model + load the pretrained backbone.
    ft_model = _model()
    key = "news_encoder.fastformer.layers.0.attention.self.query.weight"
    before = dict(ft_model.named_parameters())[key].clone()
    tuner = Finetuner(ft_model, config={"lambda1": 0.0, "lambda2": 0.0})
    tuner.load_pretrained(ckpt_dir)
    after = dict(ft_model.named_parameters())[key]
    trained = dict(model.named_parameters())[key]
    # finetune model now matches the pretrained backbone (and changed from init)
    assert torch.allclose(after, trained, atol=1e-6)
    assert not torch.allclose(after, before, atol=1e-6)


def test_finetuner_checkpoints_on_eval(tmp_path):
    torch.manual_seed(0)

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

    table = _table()
    pop = ItemPopularity({f"N{i}": 20 - i for i in range(20)})
    imprs = [
        _Impr(["N0", "N1", "N2"], [("N3", 1), ("N4", 0), ("N5", 0)]),
        _Impr(["N6", "N7"], [("N8", 1), ("N9", 0)]),
    ]
    ds = FinetuneTripletDataset(imprs, table, max_history=5, popularity=pop)
    loader = DataLoader(ds, batch_size=2, collate_fn=stack_collate)

    model = _model()
    mgr = CheckpointManager(str(tmp_path / "ft"))
    tuner = Finetuner(model, config={"epochs": 1, "eval_every": 1, "lambda1": 0.0,
                                     "lambda2": 0.0, "max_eval_impressions": 10},
                      checkpoint_manager=mgr)
    tuner.train(loader, dev_impressions=imprs, news_tokens=table)
    assert os.path.exists(tmp_path / "ft" / "epoch0" / "model.pt")
    # a best checkpoint should also exist (auc was finite)
    assert os.path.exists(tmp_path / "ft" / "best" / "model.pt")
