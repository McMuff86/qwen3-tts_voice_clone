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

### Entry Points
| Command | Description |
|---------|-------------|
| `python -m src.app` | Launch Gradio web UI |
| `qwen-tts-app` | Same (pyproject.toml script) |
| `python scripts/clone_from_file.py` | CLI voice cloning |

## Folder Structure

```
qwen3-tts_voice_clone/
├── src/                    # Core library
│   ├── __init__.py
│   ├── config.py           # Config singleton, env vars
│   ├── engine.py           # TTSEngine (lazy load, VRAM mgmt)
│   ├── audio_utils.py      # Audio I/O, combine, normalize
│   └── app.py              # Gradio web UI
├── scripts/                # Runnable scripts
│   ├── clone_from_file.py  # CLI cloning
│   ├── recorder.py         # Voice recorder
│   └── templates/          # Standalone examples
│       ├── custom_voice.py
│       ├── voice_clone.py
│       ├── voice_design.py
│       └── voice_design_then_clone.py
├── tests/                  # Pytest tests
│   ├── test_config.py
│   ├── test_engine.py
│   └── test_audio_utils.py
├── assets/
│   ├── voices/             # Voice reference audio files
│   ├── output/             # Generated audio files
│   └── samples/            # Curated example outputs
├── .env.example            # Environment variable template
├── pyproject.toml          # Project metadata & deps
└── Agents.md               # This file
```

## Legacy Files (to be removed in v0.3.0)

These root-level scripts predate the `src/` refactoring. They still work but
duplicate functionality now handled by `src.engine` + `src.app` + `scripts/`.

| Legacy Script | Replacement | Status |
|---------------|-------------|--------|
| `voice_clone_app.py` | `python -m src.app` | ⚠️ Deprecated |
| `voice_clone_from_file.py` | `scripts/clone_from_file.py` | ⚠️ Deprecated |
| `recorder.py` (root) | `scripts/recorder.py` | ⚠️ Deprecated |
| `run_tts_test.py` | `scripts/templates/custom_voice.py` | ⚠️ Deprecated |
| `template_tokenizer.py` | `scripts/templates/` (no direct replacement yet) | ⚠️ Deprecated |
| `template_voice_clone.py` | `scripts/templates/voice_clone.py` | ⚠️ Deprecated |
| `template_voice_design.py` | `scripts/templates/voice_design.py` | ⚠️ Deprecated |
| `template_voice_design_then_clone.py` | `scripts/templates/voice_design_then_clone.py` | ⚠️ Deprecated |
| `setup.py` | `pyproject.toml` (uv/pip install) | ⚠️ Deprecated |

**Cleanup plan:** Delete all legacy root scripts in a single commit once Adi confirms.
No functionality is lost – everything is covered by `src/` and `scripts/`.

## Installation

### Option 1: With uv (recommended, fast)

```bash
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
uv pip install -e ".[dev]"
```

### Option 2: With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Option 3: With Conda

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -e ".[dev]"
```

## Configuration

All settings via environment variables or `.env` file.
See `.env.example` for all available options.

```bash
cp .env.example .env
# Edit .env to your needs
```

Key variables:
- `QWEN_TTS_DEVICE` – GPU device (default: `cuda:0`)
- `QWEN_TTS_MODEL_SIZE` – `1.7B` or `0.6B`
- `QWEN_TTS_LANGUAGE` – Default generation language
- `QWEN_TTS_ATTN` – `sdpa` (default) or `flash_attention_2`

## Available Models (Downloaded)

| Model | Size | Use Case |
|-------|------|----------|
| `Qwen3-TTS-12Hz-1.7B-Base` | ~3.4 GB | Voice cloning |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~3.4 GB | 9 predefined speakers + instruction control |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~3.4 GB | Voice from text description |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | Voice cloning (lighter) |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~1.2 GB | Predefined speakers (lighter) |
| `Qwen3-TTS-Tokenizer-12Hz` | ~0.5 GB | Audio encode/decode |

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

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest tests/test_config.py  # Single file
```

Tests are designed to run without GPU/models (mock-friendly).

## TODO

### v0.3.0 – Cleanup & Polish
- [ ] Remove legacy root scripts (see table above)
- [ ] Add tokenizer template to `scripts/templates/`
- [ ] Type hints for all public functions (ruff check)
- [ ] README.md overhaul (merge README_QWEN3TTS.md content)

### Future
- [ ] Batch generation progress callback in engine
- [ ] MP3 export option
- [ ] Audio preprocessing (auto-resample reference to optimal SR)
- [ ] Model preloading option for app startup
- [ ] API server mode (FastAPI) for external integration
- [ ] Streaming generation support
