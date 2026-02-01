#!/usr/bin/env python3
"""
Voice Recorder for Voice Cloning.
==================================
Record your voice and save it as a template for voice cloning.
Files are saved to: assets/voices/

Run with: python scripts/recorder.py
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import config


def process_audio(audio_path: str | None, filename: str) -> tuple[str | None, str]:
    """Save recorded audio to the voices directory."""
    if audio_path is None:
        return None, "❌ Keine Aufnahme vorhanden."

    # Generate filename
    if not filename or not filename.strip():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}"

    # Ensure .wav extension
    if not filename.endswith(".wav"):
        filename = filename + ".wav"

    output_path = config.voices_dir / filename

    # Copy the recorded file
    shutil.copy(audio_path, str(output_path))

    return str(output_path), f"✅ Gespeichert: {output_path}"


def list_voices() -> str:
    """List all saved voice files."""
    files = sorted(config.voices_dir.glob("*.wav"))
    if not files:
        return "Keine Voice-Dateien vorhanden."
    return "\n".join(f"• {f.name}" for f in files)


with gr.Blocks(title="Voice Recorder") as demo:
    gr.Markdown("# 🎤 Voice Recorder")
    gr.Markdown(
        "Stimme aufnehmen und als Vorlage für Voice Cloning speichern.\n"
        "**Tipp:** 3-10 Sekunden klar und deutlich sprechen."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 Aufnehmen",
        )

    with gr.Row():
        filename_input = gr.Textbox(
            label="Dateiname (optional)",
            placeholder="my_voice (leer = automatisch)",
            max_lines=1,
        )

    with gr.Row():
        save_btn = gr.Button("💾 Speichern", variant="primary")

    with gr.Row():
        output_file = gr.File(label="📁 Gespeicherte Datei")
        status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Accordion("📂 Gespeicherte Voices", open=False):
        voices_list = gr.Textbox(value=list_voices(), interactive=False, lines=8)
        refresh_list_btn = gr.Button("🔄 Aktualisieren", size="sm")
        refresh_list_btn.click(fn=list_voices, outputs=voices_list)

    save_btn.click(
        fn=process_audio,
        inputs=[audio_input, filename_input],
        outputs=[output_file, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
