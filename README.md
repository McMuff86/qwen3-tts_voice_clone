# Qwen3-TTS Voice Clone

Voice cloning tools and Gradio web app built on top of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models.

Record or upload a short voice sample, then generate new speech in that voice across 10 languages.

## Features

- **Gradio Web UI** – Tabbed interface for Voice Clone, Custom Voice, and Voice Design
- **CLI Script** – Batch voice cloning from the command line with full `argparse` support
- **Voice Recorder** – Record voice samples and save them as templates
- **Engine Module** – Centralized model loading with singleton caching and VRAM management
- **Template Scripts** – Ready-to-run examples for each feature

## Requirements

- Python 3.10 – 3.12
- NVIDIA GPU with CUDA (tested with CUDA 12.1)
- ~4 GB VRAM for the 0.6B model, ~8 GB for the 1.7B model

## Setup

### 1. Create environment

```bash
# Option A: Conda
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Option B: venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Option C: Automated setup
python setup.py
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download models

```bash
pip install -U "huggingface_hub[cli]"

# Required: Tokenizer + Base model
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base

# Optional: Additional models
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### 4. (Optional) Flash Attention

```bash
pip install -U flash-attn --no-build-isolation
```

Then set `QWEN_TTS_ATTN=flash_attention_2` or update config.

## Usage

### Web UI (recommended)

```bash
python -m src.app
```

Open http://127.0.0.1:7861 – three tabs:
1. **Voice Clone** – Upload/select a voice, generate new speech
2. **Custom Voice** – Use predefined speakers (Ryan, Aiden, etc.)
3. **Voice Design** – Create voices from text descriptions

### CLI Script

```bash
# Basic usage
python scripts/clone_from_file.py --ref assets/voices/my_voice.wav "Hello world"

# With options
python scripts/clone_from_file.py \
    --ref my_voice.wav \
    --lang German \
    --transcript "Was im Audio gesagt wird" \
    --combine --pause 0.8 \
    "Erste Zeile" "Zweite Zeile" "Dritte Zeile"

# Show all options
python scripts/clone_from_file.py --help
```

### Voice Recorder

```bash
python scripts/recorder.py
```

### Template Scripts

```bash
python scripts/templates/voice_clone.py
python scripts/templates/custom_voice.py
python scripts/templates/voice_design.py
python scripts/templates/voice_design_then_clone.py
```

## Configuration

Settings can be customized via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_TTS_DEVICE` | `cuda:0` | PyTorch device |
| `QWEN_TTS_DTYPE` | `bfloat16` | Model precision |
| `QWEN_TTS_ATTN` | `sdpa` | Attention implementation |
| `QWEN_TTS_MODEL_SIZE` | `1.7B` | Default model size |
| `QWEN_TTS_LANGUAGE` | `German` | Default language |
| `QWEN_TTS_VOICES_DIR` | `assets/voices` | Voice files directory |
| `QWEN_TTS_OUTPUT_DIR` | `assets/output` | Output directory |
| `QWEN_TTS_GRADIO_PORT` | `7861` | Gradio server port |

Or create a `.env` file in the project root (requires `python-dotenv`).

## Project Structure

```
.
├── src/                            # Core modules
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration
│   ├── engine.py                   # Model loading & generation (singleton)
│   ├── audio_utils.py              # Audio processing utilities
│   └── app.py                      # Gradio web UI
├── scripts/
│   ├── clone_from_file.py          # CLI voice cloning
│   ├── recorder.py                 # Voice recorder
│   └── templates/                  # Example scripts
│       ├── voice_clone.py
│       ├── custom_voice.py
│       ├── voice_design.py
│       └── voice_design_then_clone.py
├── assets/
│   ├── voices/                     # Voice reference files
│   └── output/                     # Generated audio
├── setup.py                        # Automated setup script
├── requirements.txt                # pip dependencies
├── pyproject.toml                  # Project metadata
├── voice_clone_app.py              # [legacy] Old Gradio app
├── voice_clone_from_file.py        # [legacy] Old CLI script
├── recorder.py                     # [legacy] Old recorder
├── template_*.py                   # [legacy] Old templates
├── run_tts_test.py                 # [legacy] Custom voice test
├── voice_clone_notebook.ipynb      # Jupyter notebook
└── Qwen3-TTS-*/                    # Model directories (not in repo)
```

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Migration from v0.1

The old scripts still work but are now considered legacy. New code should use:
- `src.engine.TTSEngine` instead of direct model loading
- `src.config.config` instead of hardcoded paths
- `scripts/clone_from_file.py` instead of `voice_clone_from_file.py`
- `python -m src.app` instead of `python voice_clone_app.py`

## Credits

Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team. See [README_QWEN3TTS.md](README_QWEN3TTS.md) for the original documentation, benchmarks, and citation info.
