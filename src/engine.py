"""
TTS Engine – Model loading, caching, and generation.
=====================================================
Singleton pattern: models are loaded once and reused.

Usage:
    from src.engine import TTSEngine
    engine = TTSEngine()

    # Voice cloning
    result = engine.clone_voice(
        ref_audio="assets/voices/my_voice.wav",
        texts=["Hello world!"],
        language="English",
    )

    # Custom voice
    result = engine.generate_custom(
        texts=["Hello!"],
        speaker="Ryan",
        language="English",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config import Config, config as default_config

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    """Result of a TTS generation."""

    audio_segments: list[np.ndarray]
    sample_rate: int
    saved_files: list[Path]

    @property
    def audio(self) -> np.ndarray:
        """Return the first audio segment (convenience)."""
        return self.audio_segments[0] if self.audio_segments else np.array([])

    @property
    def combined(self) -> np.ndarray:
        """Return all segments combined with default pause."""
        from src.audio_utils import combine_audio_segments

        return combine_audio_segments(self.audio_segments, self.sample_rate)

    def combined_with_pause(self, pause_seconds: float = 0.5) -> np.ndarray:
        """Return all segments combined with a custom pause."""
        from src.audio_utils import combine_audio_segments

        return combine_audio_segments(self.audio_segments, self.sample_rate, pause_seconds)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TTSEngine:
    """Manages TTS model lifecycle and generation.

    Models are lazily loaded on first use and cached for reuse.
    Only one model is kept in memory at a time to conserve VRAM.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or default_config
        self._models: dict[str, Qwen3TTSModel] = {}
        self._current_model_key: str | None = None

    # -- model management ---------------------------------------------------

    def _load_model(self, model_type: str, size: str | None = None) -> Qwen3TTSModel:
        """Load a model, evicting any previously loaded model to save VRAM."""
        import torch
        from qwen_tts import Qwen3TTSModel

        sz = size or self.config.model_size
        key = f"{model_type}-{sz}" if model_type != "tokenizer" else "tokenizer"

        if key in self._models:
            logger.debug("Model %s already loaded, reusing.", key)
            return self._models[key]

        # Evict current model to free VRAM
        if self._current_model_key and self._current_model_key != key:
            logger.info(
                "Evicting model %s to load %s",
                self._current_model_key,
                key,
            )
            del self._models[self._current_model_key]
            self._current_model_key = None
            torch.cuda.empty_cache()

        model_id = self.config.model_id(model_type, sz)  # type: ignore[arg-type]
        logger.info("Loading model %s from %s ...", key, model_id)

        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self.config.device,
            dtype=self.config.torch_dtype,
            attn_implementation=self.config.attn_implementation,
        )

        self._models[key] = model
        self._current_model_key = key
        logger.info("Model %s loaded successfully.", key)
        return model

    def unload(self) -> None:
        """Explicitly unload all models and free VRAM."""
        import torch

        for key in list(self._models):
            del self._models[key]
        self._current_model_key = None
        torch.cuda.empty_cache()
        logger.info("All models unloaded.")

    # -- generation helpers -------------------------------------------------

    def _make_output_prefix(self, prefix: str = "output") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _save_segments(
        self,
        segments: list[np.ndarray],
        sample_rate: int,
        prefix: str,
        output_dir: Path | None = None,
    ) -> list[Path]:
        from src.audio_utils import save_audio

        out_dir = output_dir or self.config.output_dir
        saved: list[Path] = []

        for i, audio in enumerate(segments, 1):
            path = out_dir / f"{prefix}_{i}.wav"
            save_audio(audio, sample_rate, path)
            saved.append(path)
            logger.info("Saved: %s", path)

        return saved

    # -- public API ---------------------------------------------------------

    def clone_voice(
        self,
        ref_audio: str | Path | tuple[np.ndarray, int],
        texts: list[str],
        language: str | None = None,
        ref_text: str | None = None,
        output_prefix: str = "clone",
        save: bool = True,
        output_dir: Path | None = None,
    ) -> TTSResult:
        """Clone a voice from a reference audio and generate new speech.

        Parameters
        ----------
        ref_audio : str | Path | tuple
            Reference audio (path, URL, or (ndarray, sr) tuple).
        texts : list[str]
            Text(s) to generate.
        language : str | None
            Target language (default: config.default_language).
        ref_text : str | None
            Transcript of the reference audio (improves quality).
        output_prefix : str
            Prefix for saved files.
        save : bool
            Whether to save files to disk.
        output_dir : Path | None
            Custom output directory.

        Returns
        -------
        TTSResult
        """
        import torch

        lang = language or self.config.default_language
        model = self._load_model("base")

        # Create voice clone prompt
        if ref_text and ref_text.strip():
            logger.info("Creating voice clone prompt with transcript...")
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio) if isinstance(ref_audio, Path) else ref_audio,
                ref_text=ref_text.strip(),
                x_vector_only_mode=False,
            )
        else:
            logger.info("Creating voice clone prompt (x-vector only)...")
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio) if isinstance(ref_audio, Path) else ref_audio,
                x_vector_only_mode=True,
            )

        # Generate
        segments: list[np.ndarray] = []
        sample_rate: int = 24000

        try:
            for i, text in enumerate(texts, 1):
                logger.info("[%d/%d] Generating: %s", i, len(texts), text[:60])
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=voice_prompt,
                )
                sample_rate = sr
                segments.append(wavs[0])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. Try shorter text, smaller model (0.6B), "
                "or close other GPU applications."
            )

        # Save
        saved: list[Path] = []
        if save:
            prefix = self._make_output_prefix(output_prefix)
            saved = self._save_segments(segments, sample_rate, prefix, output_dir)

        return TTSResult(
            audio_segments=segments,
            sample_rate=sample_rate,
            saved_files=saved,
        )

    def generate_custom(
        self,
        texts: list[str],
        speaker: str = "Ryan",
        language: str | None = None,
        instruct: str | None = None,
        output_prefix: str = "custom",
        save: bool = True,
        output_dir: Path | None = None,
    ) -> TTSResult:
        """Generate speech with a predefined custom voice.

        Parameters
        ----------
        texts : list[str]
            Text(s) to generate.
        speaker : str
            Speaker name (Ryan, Aiden, Vivian, Serena, etc.).
        language : str | None
            Target language.
        instruct : str | None
            Style instruction (e.g. "Speak with enthusiasm").
        output_prefix : str
            Prefix for saved files.
        save : bool
            Whether to save files to disk.
        output_dir : Path | None
            Custom output directory.

        Returns
        -------
        TTSResult
        """
        import torch

        lang = language or self.config.default_language
        model = self._load_model("custom")

        segments: list[np.ndarray] = []
        sample_rate: int = 24000

        try:
            for i, text in enumerate(texts, 1):
                logger.info("[%d/%d] Generating (speaker=%s): %s", i, len(texts), speaker, text[:60])

                kwargs: dict = {
                    "text": text,
                    "language": lang,
                    "speaker": speaker,
                }
                if instruct:
                    kwargs["instruct"] = instruct

                wavs, sr = model.generate_custom_voice(**kwargs)
                sample_rate = sr
                segments.append(wavs[0])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. Try shorter text, smaller model (0.6B), "
                "or close other GPU applications."
            )

        saved: list[Path] = []
        if save:
            prefix = self._make_output_prefix(output_prefix)
            saved = self._save_segments(segments, sample_rate, prefix, output_dir)

        return TTSResult(
            audio_segments=segments,
            sample_rate=sample_rate,
            saved_files=saved,
        )

    def design_voice(
        self,
        texts: list[str],
        voice_description: str,
        language: str | None = None,
        output_prefix: str = "designed",
        save: bool = True,
        output_dir: Path | None = None,
    ) -> TTSResult:
        """Generate speech from a natural language voice description.

        Parameters
        ----------
        texts : list[str]
            Text(s) to generate.
        voice_description : str
            Description of the desired voice characteristics.
        language : str | None
            Target language.
        output_prefix : str
            Prefix for saved files.
        save : bool
            Whether to save files to disk.
        output_dir : Path | None
            Custom output directory.

        Returns
        -------
        TTSResult
        """
        import torch

        lang = language or self.config.default_language
        model = self._load_model("design")

        segments: list[np.ndarray] = []
        sample_rate: int = 24000

        try:
            for i, text in enumerate(texts, 1):
                logger.info("[%d/%d] Designing voice: %s", i, len(texts), text[:60])
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=voice_description,
                )
                sample_rate = sr
                segments.append(wavs[0])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. Try shorter text, smaller model (0.6B), "
                "or close other GPU applications."
            )

        saved: list[Path] = []
        if save:
            prefix = self._make_output_prefix(output_prefix)
            saved = self._save_segments(segments, sample_rate, prefix, output_dir)

        return TTSResult(
            audio_segments=segments,
            sample_rate=sample_rate,
            saved_files=saved,
        )
