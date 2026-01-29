"""
Qwen3-TTS Tokenizer Encode/Decode Template
==========================================
Encode audio to tokens and decode tokens back to audio.
Useful for audio compression, transport, or training purposes.
"""

import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

# Encode audio from URL
print("Encoding audio from URL...")
enc = tokenizer.encode("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav")
print(f"Encoded tokens shape: {enc.shape if hasattr(enc, 'shape') else len(enc)}")

# Decode back to audio
print("Decoding tokens back to audio...")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
print("Saved: decode_output.wav")

# Encode from local file (if you have one)
# enc_local = tokenizer.encode("your_audio_file.wav")

# Encode from numpy array
# import numpy as np
# audio_array = np.random.randn(16000)  # 1 second of audio at 16kHz
# enc_numpy = tokenizer.encode((audio_array, 16000))
