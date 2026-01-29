"""
Qwen3-TTS Voice Design then Clone Template
==========================================
Two-step workflow:
1. Design a custom voice using natural language description
2. Clone that designed voice for consistent generation across multiple utterances

This is useful when you want a specific voice style that you can reuse consistently.
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Step 1: Create a reference audio using Voice Design
print("Step 1: Designing custom voice...")

design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Design a character voice
ref_text = "Hi there! I'm so happy to help you today. Let me know what you need!"
ref_instruct = "Young female voice, approximately 25 years old, cheerful and energetic with a slight smile in the voice. Friendly and approachable customer service tone."

ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)
print("Saved reference: voice_design_reference.wav")

# Free up GPU memory
del design_model
torch.cuda.empty_cache()

# Step 2: Clone the designed voice for consistent generation
print("\nStep 2: Loading clone model and creating reusable prompt...")

clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Create reusable voice clone prompt from the designed voice
voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),  # or "voice_design_reference.wav"
    ref_text=ref_text,
)

# Generate multiple lines with consistent voice
character_lines = [
    "Welcome to TechCorp support! My name is Sarah, and I'll be assisting you today.",
    "I completely understand your frustration. Let me look into this issue right away.",
    "Great news! I've found a solution that should fix everything. Here's what we need to do.",
    "Is there anything else I can help you with today? I'm happy to assist!",
    "Thank you so much for choosing TechCorp. Have a wonderful day!"
]

print("\nStep 3: Generating character lines...")

# Generate each line
for i, line in enumerate(character_lines, 1):
    wavs, sr = clone_model.generate_voice_clone(
        text=line,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    sf.write(f"character_line_{i}.wav", wavs[0], sr)
    print(f"Saved: character_line_{i}.wav")

# Alternatively, batch generate all at once
print("\nGenerating all lines in batch...")
wavs, sr = clone_model.generate_voice_clone(
    text=character_lines,
    language=["English"] * len(character_lines),
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs, 1):
    sf.write(f"character_batch_{i}.wav", w, sr)

print(f"\nDone! Generated {len(character_lines)} character lines with consistent voice.")
