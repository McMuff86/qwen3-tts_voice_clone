"""Tests for src.engine (without GPU / model loading)."""

import numpy as np
import pytest

from src.engine import TTSResult


def test_tts_result_audio_single():
    """TTSResult.audio should return first segment."""
    segments = [np.ones(100)]
    result = TTSResult(audio_segments=segments, sample_rate=16000)
    assert np.array_equal(result.audio, segments[0])


def test_tts_result_audio_empty():
    """TTSResult.audio should return empty array when no segments."""
    result = TTSResult(audio_segments=[], sample_rate=16000)
    assert len(result.audio) == 0


def test_tts_result_combined():
    """TTSResult.combined should concatenate segments."""
    seg1 = np.ones(100, dtype=np.float32)
    seg2 = np.ones(200, dtype=np.float32) * 0.5
    result = TTSResult(audio_segments=[seg1, seg2], sample_rate=16000)

    combined = result.combined
    # 100 + pause + 200
    assert len(combined) > 300


def test_tts_result_combined_with_pause():
    """TTSResult.combined_with_pause should use custom pause."""
    seg1 = np.ones(1000, dtype=np.float32)
    seg2 = np.ones(1000, dtype=np.float32)
    result = TTSResult(audio_segments=[seg1, seg2], sample_rate=16000)

    combined = result.combined_with_pause(1.0)
    # 1000 + 16000 (1s pause) + 1000 = 18000
    assert len(combined) == 18000


def test_tts_result_summary():
    """TTSResult.summary should return a readable string."""
    result = TTSResult(
        audio_segments=[np.ones(24000)],
        sample_rate=24000,
        generation_time=2.5,
        total_duration=1.0,
    )
    summary = result.summary
    assert "1 segment" in summary
    assert "1.0s" in summary
    assert "2.5s" in summary
    assert "RTF" in summary


def test_tts_result_summary_zero_duration():
    """TTSResult.summary should handle zero duration."""
    result = TTSResult(
        audio_segments=[],
        sample_rate=24000,
        generation_time=0.0,
        total_duration=0.0,
    )
    # Should not raise
    _ = result.summary


def test_engine_init():
    """TTSEngine should initialize without loading models."""
    from src.engine import TTSEngine
    from src.config import Config

    cfg = Config(device="cpu")
    engine = TTSEngine(config=cfg)

    assert engine.loaded_model is None
    assert len(engine._models) == 0
