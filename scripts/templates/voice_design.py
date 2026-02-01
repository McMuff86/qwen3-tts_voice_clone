#!/usr/bin/env python3
"""
Voice Design Template – using the engine module.
=================================================
Create custom voices from natural language descriptions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engine import TTSEngine

engine = TTSEngine()

# Design a voice from description
result = engine.design_voice(
    texts=["Welcome to our podcast! Today we explore the world of AI."],
    voice_description=(
        "A confident, professional male voice in his 30s with a warm, "
        "engaging tone suitable for podcast hosting."
    ),
    language="English",
    output_prefix="designed_podcast",
)
print(f"Saved: {result.saved_files}")
