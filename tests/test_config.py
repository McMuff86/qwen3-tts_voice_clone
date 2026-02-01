"""Tests for src.config."""

import os
from pathlib import Path

import pytest


def test_config_defaults():
    """Config should have sensible defaults."""
    from src.config import Config

    cfg = Config()
    assert cfg.device == "cuda:0"
    assert cfg.dtype == "bfloat16"
    assert cfg.attn_implementation == "sdpa"
    assert cfg.model_size == "1.7B"
    assert cfg.default_language == "German"
    assert cfg.gradio_port == 7861


def test_config_model_id_base():
    """model_id should return correct HuggingFace repo IDs."""
    from src.config import Config

    cfg = Config()
    assert "1.7B-Base" in cfg.model_id("base", "1.7B")
    assert "0.6B-Base" in cfg.model_id("base", "0.6B")
    assert "1.7B-CustomVoice" in cfg.model_id("custom", "1.7B")
    assert "1.7B-VoiceDesign" in cfg.model_id("design", "1.7B")
    assert "Tokenizer" in cfg.model_id("tokenizer")


def test_config_model_id_prefers_local(tmp_path):
    """model_id should prefer local directory if it exists."""
    from src.config import Config

    cfg = Config(project_root=tmp_path)

    # Create a fake local model directory
    local_dir = tmp_path / "Qwen3-TTS-12Hz-1.7B-Base"
    local_dir.mkdir()

    result = cfg.model_id("base", "1.7B")
    assert result == str(local_dir)


def test_config_from_env(monkeypatch):
    """Config should read from environment variables."""
    from src.config import Config

    monkeypatch.setenv("QWEN_TTS_DEVICE", "cpu")
    monkeypatch.setenv("QWEN_TTS_DTYPE", "float16")
    monkeypatch.setenv("QWEN_TTS_MODEL_SIZE", "0.6B")
    monkeypatch.setenv("QWEN_TTS_LANGUAGE", "English")
    monkeypatch.setenv("QWEN_TTS_GRADIO_PORT", "9999")

    cfg = Config.from_env()
    assert cfg.device == "cpu"
    assert cfg.dtype == "float16"
    assert cfg.model_size == "0.6B"
    assert cfg.default_language == "English"
    assert cfg.gradio_port == 9999


def test_config_ensure_dirs(tmp_path):
    """ensure_dirs should create directories."""
    from src.config import Config

    voices = tmp_path / "voices"
    output = tmp_path / "output"
    cfg = Config(voices_dir=voices, output_dir=output)

    assert not voices.exists()
    assert not output.exists()

    cfg.ensure_dirs()

    assert voices.is_dir()
    assert output.is_dir()


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="torch not installed",
)
def test_config_torch_dtype():
    """torch_dtype should return correct torch dtype objects."""
    from src.config import Config

    import torch

    cfg = Config(dtype="bfloat16")
    assert cfg.torch_dtype == torch.bfloat16

    cfg2 = Config(dtype="float16")
    assert cfg2.torch_dtype == torch.float16

    cfg3 = Config(dtype="float32")
    assert cfg3.torch_dtype == torch.float32
