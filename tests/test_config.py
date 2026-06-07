"""Tests for newsrec.utils.config and logging."""

import os

from newsrec.utils.config import Config, deep_merge, load_config, save_config
from newsrec.utils.logging import format_metrics, setup_logger


def test_config_attr_and_dotted_access():
    cfg = Config({"model": {"hidden_size": 768, "lora": {"r": 8}}})
    assert cfg.model.hidden_size == 768
    assert cfg["model"]["lora"]["r"] == 8
    assert cfg.get("model.lora.r") == 8
    assert cfg.get("model.missing", 123) == 123


def test_config_set_creates_intermediate():
    cfg = Config()
    cfg.set("a.b.c", 5)
    assert cfg.a.b.c == 5
    assert cfg.to_dict() == {"a": {"b": {"c": 5}}}


def test_deep_merge():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    over = {"b": {"y": 20, "z": 30}, "c": 3}
    merged = deep_merge(base, over)
    assert merged == {"a": 1, "b": {"x": 1, "y": 20, "z": 30}, "c": 3}


def test_load_config_with_base_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  hidden_size: 768\n  heads: 8\ntrain:\n  lr: 0.001\n")
    child = tmp_path / "child.yaml"
    child.write_text("_base_: base.yaml\nmodel:\n  heads: 12\n")

    cfg = load_config(
        str(child),
        overrides={"train": {"epochs": 3}},
        cli_overrides=["train.lr=0.01", "model.hidden_size=256"],
    )
    assert cfg.model.hidden_size == 256  # cli override wins
    assert cfg.model.heads == 12  # child over base
    assert cfg.train.lr == 0.01  # cli coerced to float
    assert cfg.train.epochs == 3  # dict override


def test_save_and_reload_config(tmp_path):
    cfg = Config({"a": 1, "nested": {"b": 2}})
    path = tmp_path / "out" / "cfg.yaml"
    save_config(cfg, str(path))
    assert os.path.exists(path)
    reloaded = load_config(str(path))
    assert reloaded.to_dict() == {"a": 1, "nested": {"b": 2}}


def test_logger_writes_file(tmp_path):
    logger = setup_logger(
        name="test_logger",
        log_dir=str(tmp_path / "logs"),
        stage="pretrain",
        run_name="unit",
        to_console=False,
    )
    logger.info(format_metrics({"L_rec": 0.6931, "AUC": 0.5}, prefix="[step 1]"))
    log_path = logger.log_path
    assert os.path.exists(log_path)
    content = open(log_path, encoding="utf-8").read()
    assert "L_rec=0.6931" in content
    assert "pretrain_unit" in os.path.basename(log_path)
