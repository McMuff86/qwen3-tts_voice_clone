"""
Qwen3-TTS Custom Voice Generation Template
==========================================
Uses predefined speaker voices with optional instruction-based style control.

Available Speakers:
- Vivian: Bright, slightly edgy young female voice (Chinese)
- Serena: Warm, gentle young female voice (Chinese)
- Uncle_Fu: Seasoned male voice with a low, mellow timbre (Chinese)
- Dylan: Youthful Beijing male voice (Chinese - Beijing Dialect)
- Eric: Lively Chengdu male voice (Chinese - Sichuan Dialect)
- Ryan: Dynamic male voice with strong rhythmic drive (English)
- Aiden: Sunny American male voice with a clear midrange (English)
- Ono_Anna: Playful Japanese female voice (Japanese)
- Sohee: Warm Korean female voice (Korean)
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use PyTorch's native SDPA (flash_attention_2 requires flash-attn package)
)

# Single inference example
wavs, sr = model.generate_custom_voice(
    text="Hello! I'm really excited to meet you today. This is going to be a wonderful conversation.",
    language="English",  # Pass `Auto` (or omit) for auto language adaptive
    speaker="Ryan",
    instruct="Speak with enthusiasm and warmth.",  # Omit if not needed
)
sf.write("output_custom_voice.wav", wavs[0], sr)
print("Saved: output_custom_voice.wav")

# Batch inference example
wavs, sr = model.generate_custom_voice(
    text=[
        "The weather is absolutely beautiful today, perfect for a walk in the park.",
        "I can't believe you did that! This is completely unacceptable behavior."
    ],
    language=["English", "English"],
    speaker=["Aiden", "Ryan"],
    instruct=["Speak in a cheerful, relaxed tone.", "Speak with frustration and disappointment."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
print("Saved: output_custom_voice_1.wav, output_custom_voice_2.wav")
