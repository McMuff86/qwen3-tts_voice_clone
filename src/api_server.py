"""
FastAPI REST Server for Qwen3-TTS Voice Clone.
===============================================
Run with: python -m src.api_server
         uvicorn src.api_server:app --host 0.0.0.0 --port 7862

Runs alongside the Gradio UI (port 7861) on a separate port.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.config import config
from src.engine import TTSEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory for API-generated files
# ---------------------------------------------------------------------------

API_OUTPUT_DIR = config.project_root / "output" / "api"
API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Engine singleton (lazy-loaded, same pattern as Gradio app)
# ---------------------------------------------------------------------------

_engine: TTSEngine | None = None


def get_engine() -> TTSEngine:
    """Lazy-load the TTS engine on first use."""
    global _engine
    if _engine is None:
        logger.info("Initializing TTS engine for API...")
        _engine = TTSEngine()
    return _engine


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def api_response(data: Any = None, code: int = 200, error: str | None = None) -> dict:
    """Build a consistent API response envelope."""
    return {
        "data": data,
        "code": code,
        "error": error,
        "timestamp": int(time.time()),
    }


def error_response(code: int, message: str) -> JSONResponse:
    """Return a JSON error response."""
    return JSONResponse(
        status_code=code,
        content=api_response(data=None, code=code, error=message),
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CustomVoiceRequest(BaseModel):
    """Request body for custom voice generation."""
    text: str
    language: str = "English"
    speaker: str = "Ryan"
    instruct_text: str | None = None
    model_size: str = "1.7B"


class DesignVoiceRequest(BaseModel):
    """Request body for voice design generation."""
    text: str
    language: str = "English"
    voice_description: str
    model_size: str = "1.7B"


# ---------------------------------------------------------------------------
# Constants (mirrored from app.py to avoid importing Gradio)
# ---------------------------------------------------------------------------

LANGUAGES = [
    "German", "English", "French", "Spanish", "Italian",
    "Portuguese", "Russian", "Japanese", "Korean", "Chinese",
]

SPEAKERS = [
    "Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu",
    "Dylan", "Eric", "Ono_Anna", "Sohee",
]

MODEL_SIZES = ["1.7B", "0.6B"]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-TTS Voice Clone API",
    description="REST API for Qwen3-TTS voice cloning, custom voice, and voice design.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:7861",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:7861",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Health check with loaded model info."""
    engine = get_engine()
    return api_response({
        "status": "ok",
        "loaded_model": engine.loaded_model,
        "device": config.device,
        "model_size": config.model_size,
    })


@app.get("/v1/speakers")
def list_speakers() -> dict:
    """List available speakers for custom voice generation."""
    return api_response(SPEAKERS)


@app.get("/v1/languages")
def list_languages() -> dict:
    """List available languages."""
    return api_response(LANGUAGES)


@app.get("/v1/models")
def list_models() -> dict:
    """List available model sizes."""
    return api_response(MODEL_SIZES)


@app.get("/v1/voices")
def list_voices() -> dict:
    """List saved voice templates in assets/voices/."""
    voices_dir = config.voices_dir
    if not voices_dir.exists():
        return api_response([])

    voices = []
    for f in sorted(voices_dir.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
            voices.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
            })
    return api_response(voices)


@app.post("/v1/clone")
async def clone_voice(
    audio: UploadFile = File(..., description="Reference voice audio file"),
    reference_text: str = Form("", description="Transcript of the reference audio"),
    text: str = Form(..., description="Text to generate"),
    language: str = Form("English"),
    model_size: str = Form("1.7B"),
    combine: bool = Form(True),
    pause_seconds: float = Form(0.5),
) -> dict:
    """Clone a voice from a reference audio and generate new speech."""
    # Save uploaded file to temp location
    import tempfile

    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        engine = get_engine()
        result = engine.clone_voice(
            ref_audio=tmp_path,
            texts=[text],
            language=language,
            ref_text=reference_text if reference_text.strip() else None,
            model_size=model_size,
            save=False,
        )

        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clone_{timestamp}.wav"
        out_path = API_OUTPUT_DIR / filename

        if combine and len(result.audio_segments) > 1:
            from src.audio_utils import combine_audio_segments, save_audio
            combined = combine_audio_segments(
                result.audio_segments, result.sample_rate, pause_seconds
            )
            save_audio(combined, result.sample_rate, out_path)
        else:
            from src.audio_utils import save_audio
            audio_data = result.combined if len(result.audio_segments) > 1 else result.audio
            save_audio(audio_data, result.sample_rate, out_path)

        duration = len(result.combined) / result.sample_rate if result.sample_rate > 0 else 0

        return api_response({
            "audio_url": f"/v1/audio/{filename}",
            "file_path": str(out_path.resolve()),
            "duration": round(duration, 2),
            "generation_time": round(result.generation_time, 2),
            "sample_rate": result.sample_rate,
        })

    except Exception as e:
        logger.exception("Clone generation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/custom")
def generate_custom(req: CustomVoiceRequest) -> dict:
    """Generate speech with a predefined custom voice."""
    try:
        engine = get_engine()
        result = engine.generate_custom(
            texts=[req.text],
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct_text,
            model_size=req.model_size,
            save=False,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_{timestamp}.wav"
        out_path = API_OUTPUT_DIR / filename

        from src.audio_utils import save_audio
        save_audio(result.audio, result.sample_rate, out_path)

        duration = len(result.audio) / result.sample_rate if result.sample_rate > 0 else 0

        return api_response({
            "audio_url": f"/v1/audio/{filename}",
            "file_path": str(out_path.resolve()),
            "duration": round(duration, 2),
            "generation_time": round(result.generation_time, 2),
            "sample_rate": result.sample_rate,
        })

    except Exception as e:
        logger.exception("Custom voice generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/design")
def generate_design(req: DesignVoiceRequest) -> dict:
    """Generate speech from a natural language voice description."""
    try:
        engine = get_engine()
        result = engine.design_voice(
            texts=[req.text],
            voice_description=req.voice_description,
            language=req.language,
            model_size=req.model_size,
            save=False,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"design_{timestamp}.wav"
        out_path = API_OUTPUT_DIR / filename

        from src.audio_utils import save_audio
        save_audio(result.audio, result.sample_rate, out_path)

        duration = len(result.audio) / result.sample_rate if result.sample_rate > 0 else 0

        return api_response({
            "audio_url": f"/v1/audio/{filename}",
            "file_path": str(out_path.resolve()),
            "duration": round(duration, 2),
            "generation_time": round(result.generation_time, 2),
            "sample_rate": result.sample_rate,
        })

    except Exception as e:
        logger.exception("Voice design generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
) -> dict:
    """Transcribe an audio file using Whisper."""
    import tempfile

    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()

        return api_response({"text": text})

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Whisper not installed. Install with: pip install openai-whisper",
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/v1/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    """Download a generated audio file."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    file_path = API_OUTPUT_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {safe_name}")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Qwen3-TTS API server on port 7862...")
    uvicorn.run(app, host="0.0.0.0", port=7862)
