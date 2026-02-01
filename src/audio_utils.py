"""
Audio utility functions.
========================
Resampling, normalization, combining, and format conversion.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 24000  # Qwen3-TTS output sample rate


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_loudness(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to a target loudness in dB (simple RMS-based).

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (1-D float array).
    target_db : float
        Target RMS loudness in dB (default -20 dB).

    Returns
    -------
    np.ndarray
        Loudness-normalized audio.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio

    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

def combine_audio_segments(
    segments: list[np.ndarray],
    sample_rate: int,
    pause_seconds: float = 0.5,
) -> np.ndarray:
    """Combine multiple audio segments with silence in between.

    Parameters
    ----------
    segments : list[np.ndarray]
        List of audio arrays.
    sample_rate : int
        Sample rate of the audio.
    pause_seconds : float
        Silence between segments in seconds.

    Returns
    -------
    np.ndarray
        Combined audio array.
    """
    if not segments:
        return np.array([], dtype=np.float32)

    if len(segments) == 1:
        return segments[0]

    pause_samples = int(pause_seconds * sample_rate)
    silence = np.zeros(pause_samples, dtype=segments[0].dtype)

    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(silence)

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    path: str | Path,
    normalize: bool = False,
) -> Path:
    """Save audio to a WAV file.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.
    sample_rate : int
        Sample rate.
    path : str | Path
        Output file path.
    normalize : bool
        Whether to apply loudness normalization before saving.

    Returns
    -------
    Path
        The path the file was saved to.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        audio = normalize_loudness(audio)

    sf.write(str(path), audio, sample_rate)
    return path


# ---------------------------------------------------------------------------
# Loading / Resampling
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file, optionally resampling.

    Parameters
    ----------
    path : str | Path
        Path to audio file.
    target_sr : int | None
        If set, resample to this sample rate.

    Returns
    -------
    tuple[np.ndarray, int]
        Audio array and sample rate.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if target_sr and sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy (no torch dependency for utils)."""
    try:
        from scipy.signal import resample as scipy_resample

        num_samples = int(len(audio) * target_sr / orig_sr)
        return scipy_resample(audio, num_samples).astype(np.float32)
    except ImportError:
        # Fallback: simple linear interpolation
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
