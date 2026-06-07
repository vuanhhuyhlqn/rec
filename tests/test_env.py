"""Tests for newsrec.utils.env (.env loading + HF token resolution)."""

import os

from newsrec.utils.env import get_hf_token, load_dotenv


def test_load_dotenv_sets_vars(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "HUGGINGFACE_TOKEN=hf_dummy_value\n"
        'QUOTED="quoted_value"\n'
        "\n"
    )
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    assert load_dotenv(str(env)) is True
    assert os.environ["HUGGINGFACE_TOKEN"] == "hf_dummy_value"
    assert os.environ["QUOTED"] == "quoted_value"


def test_load_dotenv_missing_file():
    assert load_dotenv("/nonexistent/.env") is False


def test_load_dotenv_does_not_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "real_env_value")
    env = tmp_path / ".env"
    env.write_text("HUGGINGFACE_TOKEN=file_value\n")
    load_dotenv(str(env))  # override defaults to False
    assert os.environ["HUGGINGFACE_TOKEN"] == "real_env_value"


def test_get_hf_token_accepts_both_names(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert get_hf_token() is None
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "tok2")
    assert get_hf_token() == "tok2"
    monkeypatch.setenv("HF_TOKEN", "tok1")
    assert get_hf_token() == "tok1"  # HF_TOKEN takes precedence


def test_uploader_uses_resolved_token(monkeypatch):
    from newsrec.training.hub_uploader import HubUploader

    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "from_env")
    up = HubUploader(repo_id="user/repo", enabled=True)  # no explicit token
    assert up.enabled is True
