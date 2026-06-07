"""Tests for auto-download helper and pre-train scenario configs."""

import glob
import os

from newsrec.data.download import ensure_mind_split
from newsrec.losses.pretrain_losses import select_enabled_tasks
from newsrec.utils.config import load_config


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "newsrec", "config")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scripts")


def test_ensure_split_returns_local_when_present(tmp_path):
    d = tmp_path / "MINDsmall_train"
    d.mkdir()
    (d / "news.tsv").write_text("N1\tsports\ts\tt\ta\n")
    out = ensure_mind_split(str(d), repo_id="x/y", subfolder="train", auto_download=True)
    assert out == str(d)  # no download needed


def test_ensure_split_no_download_when_disabled(tmp_path):
    d = tmp_path / "missing"
    out = ensure_mind_split(str(d), repo_id=None, subfolder="train", auto_download=False,
                            download_dir=str(tmp_path / "dl"))
    assert out == str(d)  # returns as-is; caller raises a clear error later


def test_ensure_split_reuses_download_dir(tmp_path):
    # A prior download already populated download_dir/train -> reuse it.
    dl = tmp_path / "dataset"
    (dl / "train").mkdir(parents=True)
    (dl / "train" / "news.tsv").write_text("N1\tsports\ts\tt\ta\n")
    out = ensure_mind_split(str(tmp_path / "missing"), repo_id="x/y", subfolder="train",
                            auto_download=True, download_dir=str(dl))
    assert out == str(dl / "train")  # no network needed


def test_ensure_split_downloads_into_download_dir(tmp_path, monkeypatch):
    dl = tmp_path / "dataset"
    calls = {}

    def fake_snapshot(repo_id, repo_type, allow_patterns, token=None, local_dir=None):
        calls["local_dir"] = local_dir
        calls["allow_patterns"] = allow_patterns
        # simulate the files appearing under local_dir/train
        sub = os.path.join(local_dir, "train")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "news.tsv"), "w") as f:
            f.write("N1\tsports\ts\tt\ta\n")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    out = ensure_mind_split(str(tmp_path / "missing"), repo_id="x/y", subfolder="train",
                            auto_download=True, download_dir=str(dl))
    assert calls["local_dir"] == str(dl)            # downloaded into dataset/
    assert calls["allow_patterns"] == ["train/*"]
    assert out == str(dl / "train")


def test_all_scenario_configs_valid():
    expected = {
        "full": {"aap", "mip", "map", "sp", "bsm"},
        "item_level": {"aap", "mip", "map"},
        "sequence_level": {"mip", "sp", "bsm"},
        "attribute": {"aap", "map"},
        "aap_only": {"aap"},
        "mip_only": {"mip"},
        "map_only": {"map"},
        "sp_only": {"sp"},
        "bsm_only": {"bsm"},
    }
    for name, tasks in expected.items():
        cfg = load_config(os.path.join(CONFIG_DIR, "pretrain", f"{name}.yaml"))
        enabled = set(select_enabled_tasks(cfg.pretrain.tasks))
        assert enabled == tasks, f"{name}: {enabled} != {tasks}"
        # run_name must be unique per scenario for checkpoint namespacing
        assert cfg.run_name == f"pretrain_{name}"


def test_scenario_shell_scripts_exist_and_executable():
    for name in ("full", "item_level", "sequence_level", "attribute",
                 "aap_only", "mip_only", "map_only", "sp_only", "bsm_only"):
        path = os.path.join(SCRIPTS_DIR, f"pretrain_{name}.sh")
        assert os.path.exists(path), path
        assert os.access(path, os.X_OK), f"{path} not executable"
    # helper scripts
    for helper in ("run_all_scenarios.sh", "finetune_baseline.sh", "push_dataset.sh"):
        assert os.path.exists(os.path.join(SCRIPTS_DIR, helper))


def test_unique_run_names_across_scenarios():
    run_names = []
    for path in glob.glob(os.path.join(CONFIG_DIR, "pretrain", "*.yaml")):
        run_names.append(load_config(path).run_name)
    assert len(run_names) == len(set(run_names)), "scenario run_names must be unique"
