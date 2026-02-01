"""
Centralized configuration for Qwen3-TTS Voice Clone.
=====================================================
All paths, model settings, and defaults in one place.
Supports environment variables and .env files.

Usage:
    from src.config import config
    print(config.device)
    print(config.voices_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "base-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "custom-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "custom-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "design-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
}

ModelType = Literal["base", "custom", "design", "tokenizer"]
ModelSize = Literal["0.6B", "1.7B"]
AttnImpl = Literal["sdpa", "flash_attention_2"]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Application-wide configuration."""

    # Paths
    project_root: Path = _PROJECT_ROOT
    voices_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "assets" / "voices")
    output_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "assets" / "output")

    # Model settings
    model_size: ModelSize = "1.7B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: AttnImpl = "sdpa"

    # Generation defaults
    default_language: str = "German"
    default_pause_seconds: float = 0.5

    # Gradio
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7861

    # --- derived helpers ---------------------------------------------------

    def model_id(self, model_type: ModelType, size: ModelSize | None = None) -> str:
        """Get the HuggingFace model ID for a given type and size.

        If a local directory with the model name exists under *project_root*,
        the local path is returned instead so ``from_pretrained`` can load
        from disk without re-downloading.
        """
        sz = size or self.model_size
        if model_type == "tokenizer":
            key = "tokenizer"
        else:
            key = f"{model_type}-{sz}"

        repo_id = _MODEL_REGISTRY[key]

        # Prefer local directory (e.g. ./Qwen3-TTS-12Hz-1.7B-Base)
        local_dir = self.project_root / repo_id.split("/")[-1]
        if local_dir.is_dir():
            return str(local_dir)

        return repo_id

    @property
    def torch_dtype(self):
        """Return the actual torch dtype object."""
        import torch

        _map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return _map.get(self.dtype, torch.bfloat16)

    def ensure_dirs(self) -> None:
        """Create output and voice directories if they don't exist."""
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_cuda(self) -> dict[str, object]:
        """Check CUDA availability and return device info.

        Returns a dict with keys: available, device_name, vram_total_gb, vram_free_gb.
        If CUDA is not available, falls back to CPU and logs a warning.
        """
        import logging

        logger = logging.getLogger(__name__)
        info: dict[str, object] = {"available": False}

        try:
            import torch

            if torch.cuda.is_available():
                info["available"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_mem
                vram_free = vram_total - torch.cuda.memory_allocated(0)
                info["vram_total_gb"] = round(vram_total / 1e9, 1)
                info["vram_free_gb"] = round(vram_free / 1e9, 1)
                logger.info(
                    "CUDA available: %s (%.1f GB total, %.1f GB free)",
                    info["device_name"],
                    info["vram_total_gb"],
                    info["vram_free_gb"],
                )
            else:
                logger.warning(
                    "CUDA not available! Falling back to CPU. "
                    "Generation will be very slow."
                )
                self.device = "cpu"

        except ImportError:
            logger.warning("PyTorch not installed. Cannot check CUDA.")

        return info

    # --- factory -----------------------------------------------------------

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config from environment variables.

        Recognized variables (all optional):
            QWEN_TTS_DEVICE          – e.g. "cuda:0", "cpu"
            QWEN_TTS_DTYPE           – "bfloat16", "float16", "float32"
            QWEN_TTS_ATTN            – "sdpa" or "flash_attention_2"
            QWEN_TTS_MODEL_SIZE      – "0.6B" or "1.7B"
            QWEN_TTS_LANGUAGE        – default language
            QWEN_TTS_VOICES_DIR      – path to voices directory
            QWEN_TTS_OUTPUT_DIR      – path to output directory
            QWEN_TTS_GRADIO_PORT     – Gradio server port
        """
        # Try loading a .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv(cls().project_root / ".env")
        except ImportError:
            pass

        kwargs: dict = {}

        if v := os.getenv("QWEN_TTS_DEVICE"):
            kwargs["device"] = v
        if v := os.getenv("QWEN_TTS_DTYPE"):
            kwargs["dtype"] = v
        if v := os.getenv("QWEN_TTS_ATTN"):
            kwargs["attn_implementation"] = v  # type: ignore[arg-type]
        if v := os.getenv("QWEN_TTS_MODEL_SIZE"):
            kwargs["model_size"] = v  # type: ignore[arg-type]
        if v := os.getenv("QWEN_TTS_LANGUAGE"):
            kwargs["default_language"] = v
        if v := os.getenv("QWEN_TTS_VOICES_DIR"):
            kwargs["voices_dir"] = Path(v)
        if v := os.getenv("QWEN_TTS_OUTPUT_DIR"):
            kwargs["output_dir"] = Path(v)
        if v := os.getenv("QWEN_TTS_GRADIO_PORT"):
            kwargs["gradio_port"] = int(v)

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

config = Config.from_env()
config.ensure_dirs()
