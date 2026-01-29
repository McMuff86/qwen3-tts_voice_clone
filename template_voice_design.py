"""
Qwen3-TTS Voice Design Template
================================
Creates custom voices from natural language descriptions.
The model generates a voice based on your description of the desired characteristics.
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use PyTorch's native SDPA (flash_attention_2 requires flash-attn package)
)

# Single inference - create a voice from description
wavs, sr = model.generate_voice_design(
    text="Welcome to our podcast! Today we're going to explore the fascinating world of artificial intelligence.",
    language="English",
    instruct="A confident, professional male voice in his 30s with a warm, engaging tone suitable for podcast hosting.",
)
sf.write("output_voice_design.wav", wavs[0], sr)
print("Saved: output_voice_design.wav")

# Batch inference - multiple voice designs
wavs, sr = model.generate_voice_design(
    text=[
        "Once upon a time, in a magical forest far away, there lived a little rabbit named Fluffy.",
        "Breaking news: Scientists have made a groundbreaking discovery that could change everything we know about the universe."
    ],
    language=["English", "English"],
    instruct=[
        "A gentle, warm female voice perfect for storytelling to children. Soft and melodic with a nurturing quality.",
        "A serious, authoritative news anchor voice. Male, middle-aged, with clear diction and a sense of urgency."
    ]
)
sf.write("output_voice_design_1.wav", wavs[0], sr)
sf.write("output_voice_design_2.wav", wavs[1], sr)
print("Saved: output_voice_design_1.wav, output_voice_design_2.wav")
