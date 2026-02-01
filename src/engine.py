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

    # With progress callback
    result = engine.clone_voice(
        ...,
        on_progress=lambda i, n, text: print(f"{i}/{n}: {text}"),
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from src.config import Config, ModelType, config as default_config

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (current_index, total, current_text) -> None
ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    """Result of a TTS generation."""

    audio_segments: list[np.ndarray]
    sample_rate: int
    saved_files: list[Path] = field(default_factory=list)
    generation_time: float = 0.0  # seconds
    total_duration: float = 0.0  # audio duration in seconds

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

    @property
    def summary(self) -> str:
        """Human-readable summary of the generation."""
        duration_str = f"{self.total_duration:.1f}s"
        time_str = f"{self.generation_time:.1f}s"
        rtf = self.generation_time / self.total_duration if self.total_duration > 0 else 0
        return (
            f"{len(self.audio_segments)} segment(s), "
            f"{duration_str} audio, "
            f"generated in {time_str} "
            f"(RTF: {rtf:.2f}x)"
        )


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

    @property
    def loaded_model(self) -> str | None:
        """Return the key of the currently loaded model, or None."""
        return self._current_model_key

    def _load_model(self, model_type: ModelType, size: str | None = None) -> Qwen3TTSModel:
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
            logger.info("Evicting model %s to load %s", self._current_model_key, key)
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

    # -- internal generation core -------------------------------------------

    def _generate_loop(
        self,
        texts: list[str],
        generate_fn: Callable[[str], tuple[list[np.ndarray], int]],
        on_progress: ProgressCallback | None = None,
    ) -> tuple[list[np.ndarray], int, float]:
        """Core generation loop with timing and progress.

        Parameters
        ----------
        texts : list[str]
            Texts to generate.
        generate_fn : callable
            Function that takes a text string and returns (wavs_list, sample_rate).
        on_progress : ProgressCallback | None
            Optional callback (current_idx, total, text).

        Returns
        -------
        tuple[list[np.ndarray], int, float]
            (segments, sample_rate, elapsed_seconds)
        """
        import torch

        segments: list[np.ndarray] = []
        sample_rate: int = 24000
        t0 = time.perf_counter()

        try:
            for i, text in enumerate(texts, 1):
                if on_progress:
                    on_progress(i, len(texts), text)
                logger.info("[%d/%d] Generating: %s", i, len(texts), text[:60])

                wavs, sr = generate_fn(text)
                sample_rate = sr
                segments.append(wavs[0])

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. Try shorter text, smaller model (0.6B), "
                "or close other GPU applications."
            )

        elapsed = time.perf_counter() - t0
        return segments, sample_rate, elapsed

    def _build_result(
        self,
        segments: list[np.ndarray],
        sample_rate: int,
        elapsed: float,
        output_prefix: str,
        save: bool,
        output_dir: Path | None,
    ) -> TTSResult:
        """Build a TTSResult, optionally saving files."""
        from src.audio_utils import save_audio

        # Calculate total audio duration
        total_samples = sum(len(s) for s in segments)
        total_duration = total_samples / sample_rate if sample_rate > 0 else 0.0

        saved: list[Path] = []
        if save:
            out_dir = output_dir or self.config.output_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"{output_prefix}_{timestamp}"

            for i, audio in enumerate(segments, 1):
                path = out_dir / f"{prefix}_{i}.wav"
                save_audio(audio, sample_rate, path)
                saved.append(path)
                logger.info("Saved: %s", path)

        result = TTSResult(
            audio_segments=segments,
            sample_rate=sample_rate,
            saved_files=saved,
            generation_time=elapsed,
            total_duration=total_duration,
        )
        logger.info("Generation complete: %s", result.summary)
        return result

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
        model_size: str | None = None,
        on_progress: ProgressCallback | None = None,
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
        model_size : str | None
            Override model size ("0.6B" or "1.7B").
        on_progress : ProgressCallback | None
            Optional callback for progress updates.

        Returns
        -------
        TTSResult
        """
        lang = language or self.config.default_language
        model = self._load_model("base", model_size)

        # Normalize ref_audio
        ref = str(ref_audio) if isinstance(ref_audio, Path) else ref_audio

        # Create voice clone prompt
        if ref_text and ref_text.strip():
            logger.info("Creating voice clone prompt with transcript...")
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=ref,
                ref_text=ref_text.strip(),
                x_vector_only_mode=False,
            )
        else:
            logger.info("Creating voice clone prompt (x-vector only)...")
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=ref,
                x_vector_only_mode=True,
            )

        def gen(text: str) -> tuple[list[np.ndarray], int]:
            return model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=voice_prompt,
            )

        segments, sr, elapsed = self._generate_loop(texts, gen, on_progress)
        return self._build_result(segments, sr, elapsed, output_prefix, save, output_dir)

    def generate_custom(
        self,
        texts: list[str],
        speaker: str = "Ryan",
        language: str | None = None,
        instruct: str | None = None,
        output_prefix: str = "custom",
        save: bool = True,
        output_dir: Path | None = None,
        model_size: str | None = None,
        on_progress: ProgressCallback | None = None,
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
        model_size : str | None
            Override model size.
        on_progress : ProgressCallback | None
            Optional callback for progress updates.

        Returns
        -------
        TTSResult
        """
        lang = language or self.config.default_language
        model = self._load_model("custom", model_size)

        def gen(text: str) -> tuple[list[np.ndarray], int]:
            kwargs: dict = {"text": text, "language": lang, "speaker": speaker}
            if instruct:
                kwargs["instruct"] = instruct
            return model.generate_custom_voice(**kwargs)

        segments, sr, elapsed = self._generate_loop(texts, gen, on_progress)
        return self._build_result(segments, sr, elapsed, output_prefix, save, output_dir)

    def design_voice(
        self,
        texts: list[str],
        voice_description: str,
        language: str | None = None,
        output_prefix: str = "designed",
        save: bool = True,
        output_dir: Path | None = None,
        model_size: str | None = None,
        on_progress: ProgressCallback | None = None,
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
        model_size : str | None
            Override model size.
        on_progress : ProgressCallback | None
            Optional callback for progress updates.

        Returns
        -------
        TTSResult
        """
        lang = language or self.config.default_language
        model = self._load_model("design", model_size)

        def gen(text: str) -> tuple[list[np.ndarray], int]:
            return model.generate_voice_design(
                text=text,
                language=lang,
                instruct=voice_description,
            )

        segments, sr, elapsed = self._generate_loop(texts, gen, on_progress)
        return self._build_result(segments, sr, elapsed, output_prefix, save, output_dir)

    def design_then_clone(
        self,
        design_text: str,
        voice_description: str,
        clone_texts: list[str],
        language: str | None = None,
        output_prefix: str = "designed_clone",
        save: bool = True,
        output_dir: Path | None = None,
        model_size: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TTSResult:
        """Two-step: design a voice, then clone it for consistent output.

        Parameters
        ----------
        design_text : str
            Text to generate with the designed voice (becomes the reference).
        voice_description : str
            Description of the desired voice.
        clone_texts : list[str]
            Texts to generate with the cloned voice.
        language : str | None
            Target language.
        output_prefix : str
            Prefix for saved files.
        save : bool
            Whether to save files to disk.
        output_dir : Path | None
            Custom output directory.
        model_size : str | None
            Override model size.
        on_progress : ProgressCallback | None
            Optional callback for progress updates.

        Returns
        -------
        TTSResult
            The clone result (design reference is included as first segment).
        """
        # Step 1: Design
        if on_progress:
            on_progress(0, len(clone_texts) + 1, "Designing voice...")

        design_result = self.design_voice(
            texts=[design_text],
            voice_description=voice_description,
            language=language,
            save=False,
            model_size=model_size,
        )

        # Step 2: Unload design, clone
        self.unload()

        # Wrap progress to offset by 1
        clone_progress = None
        if on_progress:
            def clone_progress(i: int, n: int, text: str) -> None:
                on_progress(i + 1, n + 1, text)  # type: ignore[misc]

        clone_result = self.clone_voice(
            ref_audio=(design_result.audio, design_result.sample_rate),
            texts=clone_texts,
            language=language,
            ref_text=design_text,
            output_prefix=output_prefix,
            save=save,
            output_dir=output_dir,
            model_size=model_size,
            on_progress=clone_progress,
        )

        return clone_result
