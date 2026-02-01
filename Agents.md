# Qwen3-TTS Project Notes

## Architecture (v0.2.0)

### Core Modules (`src/`)
| Module | Purpose |
|--------|---------|
| `config.py` | Centralized config, env vars, model registry |
| `engine.py` | Singleton model loading, VRAM management, generation API |
| `audio_utils.py` | Resampling, normalization, combine, save/load |
| `app.py` | Gradio web UI (3 tabs: Clone, Custom, Design) |

### Scripts (`scripts/`)
| Script | Description | Run |
|--------|-------------|-----|
| `clone_from_file.py` | CLI voice cloning with argparse | `python scripts/clone_from_file.py --ref voice.wav "Text"` |
| `recorder.py` | Voice recorder UI | `python scripts/recorder.py` → :7860 |
| `templates/*.py` | Example scripts for each feature | `python scripts/templates/voice_clone.py` |

### Legacy Scripts (root)
Old scripts still work but use direct model loading. Migrate to `src.engine.TTSEngine` for new code.

| Script | Replacement |
|--------|-------------|
| `voice_clone_app.py` | `python -m src.app` |
| `voice_clone_from_file.py` | `scripts/clone_from_file.py` |
| `recorder.py` | `scripts/recorder.py` |
| `template_*.py` | `scripts/templates/*.py` |
| `run_tts_test.py` | `scripts/templates/custom_voice.py` |

## Folder Structure

```
assets/
├── voices/       # Voice reference/template audio files
├── output/       # Generated audio files
└── samples/      # Example outputs to keep
```

## Installation

### Option 1: With uv (recommended, fast)

```bash
uv venv --python 3.12
.venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

### Option 2: With Conda

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
```

### Option 3: Automated

```bash
python setup.py
```

## Environment Info

- **Python:** 3.12
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1)
- **Attention:** `sdpa` (default) or `flash_attention_2` (set via `QWEN_TTS_ATTN`)

## Configuration

All settings via environment variables or `.env`:

```bash
QWEN_TTS_DEVICE=cuda:0
QWEN_TTS_DTYPE=bfloat16
QWEN_TTS_ATTN=sdpa
QWEN_TTS_MODEL_SIZE=1.7B
QWEN_TTS_LANGUAGE=German
QWEN_TTS_VOICES_DIR=assets/voices
QWEN_TTS_OUTPUT_DIR=assets/output
QWEN_TTS_GRADIO_PORT=7861
```

## Available Models (Downloaded)

- `Qwen3-TTS-12Hz-1.7B-CustomVoice` - 9 predefined speakers with instruction control
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign` - Voice generation from descriptions
- `Qwen3-TTS-12Hz-1.7B-Base` - Voice cloning
- `Qwen3-TTS-12Hz-0.6B-CustomVoice` - Smaller custom voice model
- `Qwen3-TTS-12Hz-0.6B-Base` - Smaller base model
- `Qwen3-TTS-Tokenizer-12Hz` - Audio tokenizer

## Speakers (CustomVoice Models)

| Speaker | Language | Description |
|---------|----------|-------------|
| Ryan | English | Dynamic male voice with strong rhythmic drive |
| Aiden | English | Sunny American male voice with clear midrange |
| Vivian | Chinese | Bright, slightly edgy young female voice |
| Serena | Chinese | Warm, gentle young female voice |
| Uncle_Fu | Chinese | Seasoned male with low, mellow timbre |
| Dylan | Chinese (Beijing) | Youthful Beijing male voice |
| Eric | Chinese (Sichuan) | Lively Chengdu male voice |
| Ono_Anna | Japanese | Playful Japanese female voice |
| Sohee | Korean | Warm Korean female voice |

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## TODO

- [ ] Add basic tests (pytest)
- [ ] Type hints for all public functions
- [ ] Batch generation progress callback in engine
- [ ] MP3 export option
- [ ] Audio preprocessing (auto-resample reference to optimal SR)
- [ ] Model preloading option for app startup
- [ ] API server mode (FastAPI) for external integration
