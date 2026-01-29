# Qwen3-TTS Voice Clone

Voice cloning tools and Gradio web app built on top of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models.

Record or upload a short voice sample, then generate new speech in that voice across 10 languages.

## Features

- **Gradio Web UI** (`voice_clone_app.py`) -- upload or select a voice, enter text, get cloned audio
- **CLI Script** (`voice_clone_from_file.py`) -- batch voice cloning from the command line
- **Voice Recorder** (`recorder.py`) -- record voice samples and save them as templates
- **Template Scripts** -- ready-to-run examples for custom voice, voice design, voice clone, and tokenizer usage
- **Jupyter Notebook** (`voice_clone_notebook.ipynb`) -- interactive experimentation

## Requirements

- Python 3.10 -- 3.12
- NVIDIA GPU with CUDA (tested with CUDA 12.1)
- ~4 GB VRAM for the 0.6B model, ~8 GB for the 1.7B model

## Setup

### 1. Create environment

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv pip install -r requirements.txt
```

### 3. Download models

The models are **not** included in this repo. Download them from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"

# Tokenizer (required by all models)
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz

# Base model (voice cloning)
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base

# Optional: additional models
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### 4. (Optional) Flash Attention

For lower VRAM usage, install Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Then change `attn_implementation="sdpa"` to `attn_implementation="flash_attention_2"` in the scripts.

## Usage

### Web UI

```bash
python voice_clone_app.py
```

Open http://127.0.0.1:7861 in your browser.

1. Select a voice file from `assets/voices/` or upload one
2. Optionally provide a transcript of the reference audio
3. Enter the text to generate (one sentence per line)
4. Choose language and click **Generate Voice**

### CLI Script

Edit the configuration section in `voice_clone_from_file.py`, then run:

```bash
python voice_clone_from_file.py
```

Output files are saved to `assets/output/`.

### Template Scripts

| Script | Description |
|--------|-------------|
| `template_voice_clone.py` | Voice cloning from a reference audio |
| `template_voice_design.py` | Generate speech from a natural language voice description |
| `template_voice_design_then_clone.py` | Design a voice, then reuse it for cloning |
| `template_tokenizer.py` | Encode/decode audio with the speech tokenizer |
| `run_tts_test.py` | Custom voice generation with predefined speakers |

## Project Structure

```
.
├── voice_clone_app.py          # Gradio web UI
├── voice_clone_from_file.py    # CLI voice cloning script
├── recorder.py                 # Voice recorder utility
├── run_tts_test.py             # Custom voice test script
├── template_*.py               # Template scripts for each feature
├── voice_clone_notebook.ipynb  # Jupyter notebook
├── requirements.txt            # pip dependencies
├── pyproject.toml              # Project metadata
├── assets/
│   ├── voices/                 # Voice reference files (user-provided)
│   └── output/                 # Generated audio output
├── Qwen3-TTS-*/               # Model directories (not in repo, download separately)
└── README_QWEN3TTS.md         # Original Qwen3-TTS documentation
```

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Credits

Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team. See [README_QWEN3TTS.md](README_QWEN3TTS.md) for the original documentation, benchmarks, and citation info.
