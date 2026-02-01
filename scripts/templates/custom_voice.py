#!/usr/bin/env python3
"""
Custom Voice Template – using the engine module.
=================================================
Use predefined speakers with optional style instructions.

Available Speakers:
  Ryan, Aiden (English)
  Vivian, Serena, Uncle_Fu, Dylan, Eric (Chinese)
  Ono_Anna (Japanese), Sohee (Korean)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engine import TTSEngine

engine = TTSEngine()

# Simple generation with a predefined speaker
result = engine.generate_custom(
    texts=["Hello! I'm really excited to meet you today."],
    speaker="Ryan",
    language="English",
    instruct="Speak with enthusiasm and warmth.",
    output_prefix="custom_ryan",
)
print(f"Saved: {result.saved_files}")

# Another speaker
result = engine.generate_custom(
    texts=["The weather is absolutely beautiful today, perfect for a walk."],
    speaker="Aiden",
    language="English",
    instruct="Speak in a cheerful, relaxed tone.",
    output_prefix="custom_aiden",
)
print(f"Saved: {result.saved_files}")
