"""
Tests for the FastAPI REST Server.
===================================
Uses mocked engine to avoid GPU dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api_server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Create a mock TTSEngine that returns dummy results."""
    engine = MagicMock()
    engine.loaded_model = "base-1.7B"

    # Create a dummy TTSResult-like object
    dummy_audio = np.zeros(24000, dtype=np.float32)  # 1 second of silence
    result = MagicMock()
    result.audio_segments = [dummy_audio]
    result.audio = dummy_audio
    result.combined = dummy_audio
    result.sample_rate = 24000
    result.generation_time = 1.5
    result.total_duration = 1.0

    engine.clone_voice.return_value = result
    engine.generate_custom.return_value = result
    engine.design_voice.return_value = result

    return engine


@pytest.fixture
def patched_engine(mock_engine):
    """Patch the get_engine function to return the mock."""
    with patch("src.api_server.get_engine", return_value=mock_engine):
        yield mock_engine


# ---------------------------------------------------------------------------
# Health & list endpoints
# ---------------------------------------------------------------------------

class TestHealthAndLists:
    def test_health(self, patched_engine):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["status"] == "ok"
        assert body["data"]["loaded_model"] == "base-1.7B"
        assert body["error"] is None
        assert "timestamp" in body

    def test_speakers(self):
        resp = client.get("/v1/speakers")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert "Ryan" in data

    def test_languages(self):
        resp = client.get("/v1/languages")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "English" in data
        assert "German" in data

    def test_models(self):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "1.7B" in data
        assert "0.6B" in data

    def test_voices(self):
        resp = client.get("/v1/voices")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert isinstance(body["data"], list)


# ---------------------------------------------------------------------------
# Clone endpoint
# ---------------------------------------------------------------------------

class TestClone:
    def test_clone_success(self, patched_engine):
        with patch("src.audio_utils.save_audio"):
            resp = client.post(
                "/v1/clone",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
                data={
                    "text": "Hello world",
                    "reference_text": "Reference text",
                    "language": "English",
                },
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "audio_url" in data
        assert data["audio_url"].startswith("/v1/audio/")
        assert data["sample_rate"] == 24000

    def test_clone_missing_text(self):
        resp = client.post(
            "/v1/clone",
            files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
        )
        assert resp.status_code == 422

    def test_clone_missing_audio(self):
        resp = client.post(
            "/v1/clone",
            data={"text": "Hello"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Custom voice endpoint
# ---------------------------------------------------------------------------

class TestCustom:
    def test_custom_success(self, patched_engine):
        with patch("src.audio_utils.save_audio"):
            resp = client.post(
                "/v1/custom",
                json={
                    "text": "Hello world",
                    "language": "English",
                    "speaker": "Ryan",
                },
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "audio_url" in data

    def test_custom_missing_text(self):
        resp = client.post("/v1/custom", json={"language": "English"})
        assert resp.status_code == 422

    def test_custom_defaults(self, patched_engine):
        with patch("src.audio_utils.save_audio"):
            resp = client.post("/v1/custom", json={"text": "Test"})
        assert resp.status_code == 200
        # Verify defaults were used
        call_kwargs = patched_engine.generate_custom.call_args
        assert call_kwargs.kwargs["speaker"] == "Ryan"
        assert call_kwargs.kwargs["language"] == "English"


# ---------------------------------------------------------------------------
# Design voice endpoint
# ---------------------------------------------------------------------------

class TestDesign:
    def test_design_success(self, patched_engine):
        with patch("src.audio_utils.save_audio"):
            resp = client.post(
                "/v1/design",
                json={
                    "text": "Hello world",
                    "voice_description": "A warm male voice",
                },
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "audio_url" in data

    def test_design_missing_description(self):
        resp = client.post("/v1/design", json={"text": "Hello"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Audio download endpoint
# ---------------------------------------------------------------------------

class TestAudioDownload:
    def test_audio_not_found(self):
        resp = client.get("/v1/audio/nonexistent.wav")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self):
        resp = client.get("/v1/audio/../../etc/passwd")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Transcribe endpoint
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_transcribe_missing_audio(self):
        resp = client.post("/v1/transcribe")
        assert resp.status_code == 422

    def test_transcribe_success(self):
        import sys

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world"}

        # Ensure whisper module exists for patching even if not installed
        fake_whisper = MagicMock()
        fake_whisper.load_model.return_value = mock_model
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            resp = client.post(
                "/v1/transcribe",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "Hello world"
