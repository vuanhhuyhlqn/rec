"""Task 13: tiny end-to-end smoke test (pretrain -> finetune -> eval)."""

import os

import pytest

from newsrec.utils.config import load_config


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "newsrec", "config")


def _have_data(train_dir, dev_dir):
    return os.path.exists(os.path.join(train_dir, "news.tsv")) and \
        os.path.exists(os.path.join(dev_dir, "news.tsv"))


def test_configs_load_and_inherit_base():
    # presets correctly inherit base.yaml
    pre = load_config(os.path.join(CONFIG_DIR, "pretrain", "full.yaml"))
    assert pre.run_name == "pretrain_full"
    assert pre.model.plm.model_name == "bert-base-uncased"  # from base
    assert "aap" in pre.pretrain.tasks

    ft = load_config(os.path.join(CONFIG_DIR, "finetune", "paac.yaml"))
    assert ft.finetune.lambda1 == 0.1
    assert ft.data.max_history == 50  # from base


def test_smoke_end_to_end(tmp_path, train_dir, dev_dir):
    if not _have_data(train_dir, dev_dir):
        pytest.skip("MINDsmall data not present")

    from newsrec.scripts.run_finetune import run_finetune
    from newsrec.scripts.run_pretrain import run_pretrain

    ckpt_root = str(tmp_path / "checkpoints")

    # --- Pretrain (tiny) ---
    pre_cfg = load_config(
        os.path.join(CONFIG_DIR, "smoke", "pretrain.yaml"),
        cli_overrides=[
            f"data.train_dir={train_dir}",
            f"data.dev_dir={dev_dir}",
            f"checkpoint.dir={ckpt_root}",
            "logging.log_dir=" + str(tmp_path / "logs"),
        ],
    )
    run_pretrain(pre_cfg)
    best_dir = os.path.join(ckpt_root, "smoke", "pretrain", "best")
    assert os.path.exists(os.path.join(best_dir, "model.pt"))

    # --- Finetune (tiny), loading the pretrained backbone ---
    ft_cfg = load_config(
        os.path.join(CONFIG_DIR, "smoke", "finetune.yaml"),
        cli_overrides=[
            f"data.train_dir={train_dir}",
            f"data.dev_dir={dev_dir}",
            f"checkpoint.dir={ckpt_root}",
            "logging.log_dir=" + str(tmp_path / "logs"),
            f"finetune.pretrained_ckpt={best_dir}",
        ],
    )
    run_finetune(ft_cfg)
    assert os.path.exists(os.path.join(ckpt_root, "smoke", "finetune", "epoch0", "model.pt"))
