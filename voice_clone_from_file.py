"""
Voice Clone from Audio File
============================
Clone a voice from your own audio file and generate new speech.

Usage:
1. Place your voice reference audio in: assets/voices/
2. Set REFERENCE_AUDIO to your audio filename
3. Set REFERENCE_TEXT to the transcript (optional but recommended)
4. Set TEXT_TO_GENERATE to what you want the cloned voice to say
5. Run the script - outputs go to: assets/output/
"""

import os
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# =============================================================================
# PATHS
# =============================================================================
VOICES_DIR = "assets/voices"    # Voice reference/template files
OUTPUT_DIR = "assets/output"    # Generated audio files

# Ensure directories exist
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Filename of your reference audio (place it in assets/voices/)
REFERENCE_AUDIO = "my_voice_test.wav"  # Change this to your audio file

# Transcript of what is said in the reference audio (for better quality)
# Leave empty "" if you don't know it (quality will be slightly lower)
REFERENCE_TEXT = ""

# The text you want to generate with the cloned voice
TEXT_TO_GENERATE = [
    "Hello, this is a test of voice cloning technology.",
    "I can say anything you want me to say in this voice.",
    "Pretty amazing, right? The future is here!",
]

# Output file prefix (files saved as: assets/output/{prefix}_1.wav, etc.)
OUTPUT_PREFIX = "cloned"

# Combine all audio into one file? (in addition to individual files)
COMBINE_INTO_ONE = True

# Pause between sentences in combined file (in seconds)
PAUSE_BETWEEN_SENTENCES = 0.5

# =============================================================================
# SCRIPT - Don't edit below unless you know what you're doing
# =============================================================================

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Build full path to reference audio
ref_audio_path = os.path.join(VOICES_DIR, REFERENCE_AUDIO)

if not os.path.exists(ref_audio_path):
    print(f"ERROR: Reference audio not found: {ref_audio_path}")
    print(f"       Please place your voice file in: {VOICES_DIR}/")
    exit(1)

print(f"Creating voice clone from: {ref_audio_path}")

# Create reusable voice clone prompt
if REFERENCE_TEXT:
    print("Using transcript for better quality cloning...")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=REFERENCE_TEXT,
        x_vector_only_mode=False,
    )
else:
    print("No transcript provided, using x-vector only mode...")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        x_vector_only_mode=True,
    )

print(f"\nGenerating {len(TEXT_TO_GENERATE)} audio files...")

# Handle single string or list
if isinstance(TEXT_TO_GENERATE, str):
    TEXT_TO_GENERATE = [TEXT_TO_GENERATE]

# Collect all audio for combining later
all_audio = []
sample_rate = None

# Generate each text
for i, text in enumerate(TEXT_TO_GENERATE, 1):
    print(f"  [{i}/{len(TEXT_TO_GENERATE)}] Generating: {text[:50]}...")
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",  # Change if needed: German, French, Spanish, etc.
        voice_clone_prompt=voice_prompt,
    )
    
    sample_rate = sr
    all_audio.append(wavs[0])
    
    output_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{i}.wav")
    sf.write(output_file, wavs[0], sr)
    print(f"           Saved: {output_file}")

# Combine all audio into one file if requested
if COMBINE_INTO_ONE and len(all_audio) > 1:
    print("\nCombining all audio into one file...")
    
    # Create silence for pause between sentences
    pause_samples = int(PAUSE_BETWEEN_SENTENCES * sample_rate)
    silence = np.zeros(pause_samples, dtype=all_audio[0].dtype)
    
    # Combine with pauses
    combined = []
    for i, audio in enumerate(all_audio):
        combined.append(audio)
        if i < len(all_audio) - 1:  # Don't add pause after last segment
            combined.append(silence)
    
    combined_audio = np.concatenate(combined)
    combined_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_combined.wav")
    sf.write(combined_file, combined_audio, sample_rate)
    print(f"           Saved: {combined_file}")

print(f"\nDone! Generated {len(TEXT_TO_GENERATE)} audio files.")
