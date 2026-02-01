"""Tests for src.audio_utils."""

import numpy as np
import pytest
from pathlib import Path


# -- Pure numpy tests (no external deps) ------------------------------------

def test_normalize_loudness():
    """normalize_loudness should adjust RMS to target."""
    from src.audio_utils import normalize_loudness

    audio = np.ones(1000, dtype=np.float32) * 0.5
    normalized = normalize_loudness(audio, target_db=-20.0)

    rms = np.sqrt(np.mean(normalized ** 2))
    target_rms = 10 ** (-20.0 / 20.0)
    assert abs(rms - target_rms) < 0.01


def test_normalize_loudness_silent():
    """normalize_loudness should handle silent audio gracefully."""
    from src.audio_utils import normalize_loudness

    silent = np.zeros(1000, dtype=np.float32)
    result = normalize_loudness(silent)
    assert np.all(result == 0)


def test_combine_audio_segments():
    """combine_audio_segments should concatenate with silence."""
    from src.audio_utils import combine_audio_segments

    sr = 16000
    seg1 = np.ones(1000, dtype=np.float32)
    seg2 = np.ones(2000, dtype=np.float32) * 0.5

    combined = combine_audio_segments([seg1, seg2], sr, pause_seconds=0.5)

    pause_samples = int(0.5 * sr)
    expected_length = 1000 + pause_samples + 2000
    assert len(combined) == expected_length
    assert np.all(combined[1000:1000 + pause_samples] == 0)


def test_combine_single_segment():
    """combine_audio_segments with one segment should return it as-is."""
    from src.audio_utils import combine_audio_segments

    seg = np.ones(1000, dtype=np.float32)
    result = combine_audio_segments([seg], 16000)
    assert np.array_equal(result, seg)


def test_combine_empty():
    """combine_audio_segments with empty list should return empty array."""
    from src.audio_utils import combine_audio_segments

    result = combine_audio_segments([], 16000)
    assert len(result) == 0


# -- Tests requiring soundfile ----------------------------------------------

@pytest.fixture
def _needs_soundfile():
    pytest.importorskip("soundfile", reason="soundfile not installed")


@pytest.mark.usefixtures("_needs_soundfile")
def test_save_audio(tmp_path):
    """save_audio should write a valid WAV file."""
    import soundfile as sf
    from src.audio_utils import save_audio

    audio = np.random.randn(16000).astype(np.float32)
    path = save_audio(audio, 16000, tmp_path / "test.wav")

    assert path.exists()
    assert path.suffix == ".wav"

    data, sr = sf.read(str(path))
    assert sr == 16000
    assert len(data) == 16000


@pytest.mark.usefixtures("_needs_soundfile")
def test_save_audio_creates_dirs(tmp_path):
    """save_audio should create parent directories."""
    from src.audio_utils import save_audio

    audio = np.random.randn(100).astype(np.float32)
    nested = tmp_path / "a" / "b" / "c" / "test.wav"
    path = save_audio(audio, 16000, nested)
    assert path.exists()


@pytest.mark.usefixtures("_needs_soundfile")
def test_load_audio(tmp_path):
    """load_audio should read WAV files."""
    import soundfile as sf
    from src.audio_utils import load_audio

    audio = np.random.randn(16000).astype(np.float32)
    test_file = tmp_path / "test.wav"
    sf.write(str(test_file), audio, 16000)

    loaded, sr = load_audio(test_file)
    assert sr == 16000
    assert len(loaded) == 16000


@pytest.mark.usefixtures("_needs_soundfile")
def test_load_audio_stereo_to_mono(tmp_path):
    """load_audio should convert stereo to mono."""
    import soundfile as sf
    from src.audio_utils import load_audio

    stereo = np.random.randn(16000, 2).astype(np.float32)
    test_file = tmp_path / "stereo.wav"
    sf.write(str(test_file), stereo, 16000)

    loaded, sr = load_audio(test_file)
    assert loaded.ndim == 1
    assert len(loaded) == 16000
