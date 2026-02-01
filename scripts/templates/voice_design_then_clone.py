#!/usr/bin/env python3
"""
Voice Design then Clone Template – using the engine module.
============================================================
Two-step workflow:
1. Design a custom voice using natural language description
2. Clone that designed voice for consistent generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.engine import TTSEngine
from src.audio_utils import save_audio

engine = TTSEngine()

# Step 1: Design a voice
print("Step 1: Designing custom voice...")
ref_result = engine.design_voice(
    texts=["Hi there! I'm so happy to help you today. Let me know what you need!"],
    voice_description=(
        "Young female voice, approximately 25 years old, cheerful and energetic "
        "with a slight smile in the voice. Friendly customer service tone."
    ),
    language="English",
    output_prefix="design_ref",
)
print(f"Reference saved: {ref_result.saved_files[0]}")

# Step 2: Clone that voice for consistent generation
print("\nStep 2: Cloning designed voice for consistent output...")

# Unload design model, load base model
engine.unload()

result = engine.clone_voice(
    ref_audio=ref_result.saved_files[0],
    texts=[
        "Welcome to TechCorp support! My name is Sarah.",
        "I completely understand your frustration. Let me look into this.",
        "Great news! I've found a solution. Here's what we need to do.",
        "Thank you for choosing TechCorp. Have a wonderful day!",
    ],
    language="English",
    ref_text="Hi there! I'm so happy to help you today. Let me know what you need!",
    output_prefix="character_sarah",
)

# Save combined version
combined = result.combined_with_pause(0.8)
combined_path = config.output_dir / "character_sarah_combined.wav"
save_audio(combined, result.sample_rate, combined_path)

print(f"\nSaved {len(result.saved_files)} individual files + combined: {combined_path}")
