"""
Qwen3-TTS Voice Clone Template
===============================
Clone a voice from a reference audio sample and generate new speech.
Requires a reference audio file and its transcript for best results.
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use PyTorch's native SDPA (flash_attention_2 requires flash-attn package)
)

# Reference audio for voice cloning
# Can be: local file path, URL, base64 string, or (numpy_array, sample_rate) tuple
ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

# Single inference - clone voice and generate new text
wavs, sr = model.generate_voice_clone(
    text="Hello everyone! Today I want to share something really exciting with you. Let's dive right in!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
print("Saved: output_voice_clone.wav")

# Reusable prompt for multiple generations (more efficient)
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,  # Set to True if you don't have the transcript (lower quality)
)

# Generate multiple sentences with the same cloned voice
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Technology has transformed the way we live, work, and communicate.",
    "I'm really looking forward to our meeting next week. See you then!"
]

for i, sentence in enumerate(sentences):
    wavs, sr = model.generate_voice_clone(
        text=sentence,
        language="English",
        voice_clone_prompt=prompt_items,
    )
    sf.write(f"output_clone_batch_{i+1}.wav", wavs[0], sr)
    print(f"Saved: output_clone_batch_{i+1}.wav")

# Batch generation (all at once)
wavs, sr = model.generate_voice_clone(
    text=sentences,
    language=["English"] * len(sentences),
    voice_clone_prompt=prompt_items,
)
for i, w in enumerate(wavs):
    sf.write(f"output_clone_all_{i+1}.wav", w, sr)
print(f"Saved: output_clone_all_1.wav through output_clone_all_{len(wavs)}.wav")
