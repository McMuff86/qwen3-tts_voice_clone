"""
Voice Clone Gradio App (refactored).
=====================================
Web interface for voice cloning using Qwen3-TTS.

Run with: python -m src.app
Then open: http://127.0.0.1:7861
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import numpy as np

from src.config import config
from src.engine import TTSEngine
from src.audio_utils import combine_audio_segments, save_audio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine (lazy-loaded)
# ---------------------------------------------------------------------------

_engine: TTSEngine | None = None


def get_engine() -> TTSEngine:
    """Lazy-load the TTS engine on first use."""
    global _engine
    if _engine is None:
        logger.info("Initializing TTS engine...")
        _engine = TTSEngine()
    return _engine


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

LANGUAGES = [
    "German", "English", "French", "Spanish", "Italian",
    "Portuguese", "Russian", "Japanese", "Korean", "Chinese",
]

SPEAKERS = [
    "Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu",
    "Dylan", "Eric", "Ono_Anna", "Sohee",
]


def get_voice_files() -> list[str]:
    """Get list of voice files in the voices directory."""
    if not config.voices_dir.exists():
        return []
    files = [
        f.name for f in config.voices_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def preview_voice(voice_file: str | None) -> str | None:
    """Return path to voice file for preview playback."""
    if voice_file:
        path = config.voices_dir / voice_file
        if path.exists():
            return str(path)
    return None


# ---------------------------------------------------------------------------
# Clone tab
# ---------------------------------------------------------------------------

def clone_voice(
    voice_file_dropdown: str | None,
    voice_file_upload: str | None,
    reference_text: str,
    text_to_generate: str,
    language: str,
    combine_audio: bool,
    pause_seconds: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[tuple[int, np.ndarray] | None, str | None, str]:
    """Main voice cloning function."""

    # Determine voice file
    if voice_file_upload is not None:
        ref_audio_path = voice_file_upload
        voice_name = "uploaded"
    elif voice_file_dropdown:
        ref_audio_path = str(config.voices_dir / voice_file_dropdown)
        voice_name = Path(voice_file_dropdown).stem
    else:
        return None, None, "❌ Bitte eine Voice-Datei auswählen oder hochladen!"

    if not Path(ref_audio_path).exists():
        return None, None, f"❌ Datei nicht gefunden: {ref_audio_path}"

    # Parse text
    texts = [t.strip() for t in text_to_generate.strip().split("\n") if t.strip()]
    if not texts:
        return None, None, "❌ Bitte Text eingeben!"

    status_lines: list[str] = []

    try:
        engine = get_engine()

        # Progress tracking
        for i in range(len(texts)):
            progress((i) / len(texts), f"Generiere {i + 1}/{len(texts)}...")

        result = engine.clone_voice(
            ref_audio=ref_audio_path,
            texts=texts,
            language=language,
            ref_text=reference_text if reference_text and reference_text.strip() else None,
            output_prefix=voice_name,
            save=True,
        )

        for f in result.saved_files:
            status_lines.append(f"✅ Gespeichert: {f.name}")

        # Combine if requested
        if combine_audio and len(result.audio_segments) > 1:
            combined = result.combined_with_pause(pause_seconds)
            combined_path = config.output_dir / f"{voice_name}_combined.wav"
            save_audio(combined, result.sample_rate, combined_path)
            status_lines.append(f"✅ Kombiniert: {combined_path.name}")
            return (result.sample_rate, combined), str(combined_path), "\n".join(status_lines)

        # Single file
        return (
            (result.sample_rate, result.audio),
            str(result.saved_files[0]) if result.saved_files else None,
            "\n".join(status_lines),
        )

    except RuntimeError as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        logger.exception("Unexpected error during voice cloning")
        return None, None, f"❌ Unerwarteter Fehler: {e}"


# ---------------------------------------------------------------------------
# Custom voice tab
# ---------------------------------------------------------------------------

def generate_custom_voice(
    text_to_generate: str,
    language: str,
    speaker: str,
    instruct: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[tuple[int, np.ndarray] | None, str | None, str]:
    """Generate speech with a predefined speaker."""

    texts = [t.strip() for t in text_to_generate.strip().split("\n") if t.strip()]
    if not texts:
        return None, None, "❌ Bitte Text eingeben!"

    try:
        engine = get_engine()
        progress(0.5, "Generiere...")

        result = engine.generate_custom(
            texts=texts,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct and instruct.strip() else None,
            output_prefix=f"custom_{speaker.lower()}",
        )

        status = "\n".join(f"✅ Gespeichert: {f.name}" for f in result.saved_files)

        if len(result.audio_segments) > 1:
            combined = result.combined
            return (result.sample_rate, combined), str(result.saved_files[0]), status

        return (result.sample_rate, result.audio), str(result.saved_files[0]), status

    except RuntimeError as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        logger.exception("Unexpected error during custom voice generation")
        return None, None, f"❌ Unerwarteter Fehler: {e}"


# ---------------------------------------------------------------------------
# Voice design tab
# ---------------------------------------------------------------------------

def design_and_generate(
    text_to_generate: str,
    language: str,
    voice_description: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[tuple[int, np.ndarray] | None, str | None, str]:
    """Design a voice from description and generate speech."""

    texts = [t.strip() for t in text_to_generate.strip().split("\n") if t.strip()]
    if not texts:
        return None, None, "❌ Bitte Text eingeben!"

    if not voice_description or not voice_description.strip():
        return None, None, "❌ Bitte eine Stimmbeschreibung eingeben!"

    try:
        engine = get_engine()
        progress(0.5, "Generiere...")

        result = engine.design_voice(
            texts=texts,
            voice_description=voice_description.strip(),
            language=language,
            output_prefix="designed",
        )

        status = "\n".join(f"✅ Gespeichert: {f.name}" for f in result.saved_files)
        return (result.sample_rate, result.audio), str(result.saved_files[0]), status

    except RuntimeError as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        logger.exception("Unexpected error during voice design")
        return None, None, f"❌ Unerwarteter Fehler: {e}"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(title="Qwen3-TTS Voice Clone", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Qwen3-TTS Voice Clone")
        gr.Markdown("Voice Cloning, Custom Voices und Voice Design mit Qwen3-TTS.")

        with gr.Tabs():
            # ── Tab 1: Voice Cloning ──────────────────────────────────────
            with gr.TabItem("🔊 Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Stimme auswählen")

                        voice_dropdown = gr.Dropdown(
                            choices=get_voice_files(),
                            label="Voice aus assets/voices/",
                            info="Gespeicherte Stimmvorlagen",
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Aktualisieren", size="sm")
                            preview_audio = gr.Audio(
                                label="Vorschau",
                                type="filepath",
                                interactive=False,
                            )

                        gr.Markdown("**— ODER —**")

                        voice_upload = gr.Audio(
                            label="Voice-Datei hochladen",
                            type="filepath",
                            sources=["upload"],
                        )

                        reference_text = gr.Textbox(
                            label="Transkript (optional)",
                            placeholder="Was wird im Referenz-Audio gesagt...",
                            info="Verbessert die Qualität deutlich!",
                            lines=2,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Text generieren")

                        clone_text = gr.Textbox(
                            label="Text",
                            placeholder="Text eingeben...\nJede Zeile wird einzeln generiert.",
                            lines=6,
                            value="Hallo, das ist ein Test der Stimmenklonungs-Technologie.\nIch kann alles sagen, was du möchtest.",
                        )
                        clone_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value=config.default_language,
                            label="Sprache",
                        )
                        with gr.Row():
                            combine_check = gr.Checkbox(label="Kombinieren", value=True)
                            pause_slider = gr.Slider(0, 2, value=0.5, step=0.1, label="Pause (Sek)")

                clone_btn = gr.Button("🎙️ Voice klonen", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        clone_audio_out = gr.Audio(label="Generiertes Audio", type="numpy")
                        clone_file_out = gr.File(label="Download")
                    with gr.Column():
                        clone_status = gr.Textbox(label="Status", lines=8, interactive=False)

                # Events
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_voice_files(), value=None),
                    outputs=voice_dropdown,
                )
                voice_dropdown.change(
                    fn=preview_voice,
                    inputs=voice_dropdown,
                    outputs=preview_audio,
                )
                clone_btn.click(
                    fn=clone_voice,
                    inputs=[
                        voice_dropdown, voice_upload, reference_text,
                        clone_text, clone_language, combine_check, pause_slider,
                    ],
                    outputs=[clone_audio_out, clone_file_out, clone_status],
                )

            # ── Tab 2: Custom Voice ───────────────────────────────────────
            with gr.TabItem("🗣️ Custom Voice"):
                with gr.Row():
                    with gr.Column():
                        custom_speaker = gr.Dropdown(
                            choices=SPEAKERS,
                            value="Ryan",
                            label="Sprecher",
                        )
                        custom_instruct = gr.Textbox(
                            label="Stil-Anweisung (optional)",
                            placeholder="z.B. 'Speak with enthusiasm and warmth.'",
                            lines=2,
                        )
                    with gr.Column():
                        custom_text = gr.Textbox(
                            label="Text",
                            placeholder="Text eingeben...",
                            lines=4,
                            value="Hello, I'm excited to demonstrate this voice synthesis technology.",
                        )
                        custom_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="English",
                            label="Sprache",
                        )

                custom_btn = gr.Button("🎙️ Generieren", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        custom_audio_out = gr.Audio(label="Generiertes Audio", type="numpy")
                        custom_file_out = gr.File(label="Download")
                    with gr.Column():
                        custom_status = gr.Textbox(label="Status", lines=6, interactive=False)

                custom_btn.click(
                    fn=generate_custom_voice,
                    inputs=[custom_text, custom_language, custom_speaker, custom_instruct],
                    outputs=[custom_audio_out, custom_file_out, custom_status],
                )

            # ── Tab 3: Voice Design ───────────────────────────────────────
            with gr.TabItem("✨ Voice Design"):
                with gr.Row():
                    with gr.Column():
                        design_description = gr.Textbox(
                            label="Stimmbeschreibung",
                            placeholder="z.B. 'A warm, deep male voice in his 40s with a slight German accent.'",
                            lines=3,
                        )
                    with gr.Column():
                        design_text = gr.Textbox(
                            label="Text",
                            placeholder="Text eingeben...",
                            lines=4,
                            value="Welcome to our presentation. Today I want to share something exciting with you.",
                        )
                        design_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="English",
                            label="Sprache",
                        )

                design_btn = gr.Button("✨ Voice designen", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        design_audio_out = gr.Audio(label="Generiertes Audio", type="numpy")
                        design_file_out = gr.File(label="Download")
                    with gr.Column():
                        design_status = gr.Textbox(label="Status", lines=6, interactive=False)

                design_btn.click(
                    fn=design_and_generate,
                    inputs=[design_text, design_language, design_description],
                    outputs=[design_audio_out, design_file_out, design_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = build_app()
    app.launch(server_name=config.gradio_host, server_port=config.gradio_port)


if __name__ == "__main__":
    main()
