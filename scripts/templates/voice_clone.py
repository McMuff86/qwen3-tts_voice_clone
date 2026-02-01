#!/usr/bin/env python3
"""
Voice Clone Template – using the engine module.
================================================
Clone a voice from a reference audio sample and generate new speech.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engine import TTSEngine

engine = TTSEngine()

# --- Option 1: Simple clone ---
result = engine.clone_voice(
    ref_audio="assets/voices/my_voice.wav",
    texts=["Hello everyone! This is a voice cloning test."],
    language="English",
    output_prefix="clone_simple",
)
print(f"Saved: {result.saved_files}")


# --- Option 2: With transcript (better quality) ---
result = engine.clone_voice(
    ref_audio="assets/voices/my_voice.wav",
    texts=[
        "The quick brown fox jumps over the lazy dog.",
        "Technology has transformed the way we live and work.",
    ],
    language="English",
    ref_text="What is said in the reference audio goes here.",
    output_prefix="clone_transcript",
)
print(f"Saved: {result.saved_files}")


# --- Option 3: Get combined audio ---
combined = result.combined_with_pause(pause_seconds=0.8)
from src.audio_utils import save_audio
save_audio(combined, result.sample_rate, "assets/output/clone_combined.wav")
