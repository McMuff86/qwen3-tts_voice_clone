# Qwen3-TTS Project Notes

## Folder Structure

```
assets/
├── voices/       # Voice reference/template audio files (your voice samples)
├── output/       # Generated audio files
└── samples/      # Example outputs to keep
```

## Installation

### Option 1: With uv (recommended, fast)

```bash
# Install uv (if not installed)
pip install uv

# Create virtual environment and install
uv venv --python 3.12
.venv\Scripts\activate  # Windows
uv pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name qwen3-tts --display-name "Python (qwen3-tts)"
```

### Option 2: With Conda

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
python -m ipykernel install --user --name qwen3-tts --display-name "Python (qwen3-tts)"
```

### Option 3: With pip

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Environment Info

- **Python:** 3.12
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1)
- **Attention Implementation:** `sdpa` (PyTorch's native Scaled Dot Product Attention)

### Note on Flash Attention
Flash Attention 2 is not installed. To enable it:
1. Upgrade PyTorch to CUDA 12.4: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
2. Install prebuilt wheel from HuggingFace
3. Change `attn_implementation` to `"flash_attention_2"` in scripts

Current setup uses `sdpa` which works well without additional dependencies.

## Apps & Scripts

### Main Apps
| Script | Description | Run |
|--------|-------------|-----|
| `voice_clone_app.py` | Gradio Web UI for voice cloning | `python voice_clone_app.py` → http://127.0.0.1:7861 |
| `voice_clone_notebook.ipynb` | Jupyter Notebook version | Open in VS Code/Jupyter |
| `voice_clone_from_file.py` | CLI script for voice cloning | `python voice_clone_from_file.py` |
| `recorder.py` | Record voice templates | `python recorder.py` → http://127.0.0.1:7860 |

### Template Scripts
| Script | Model | Description |
|--------|-------|-------------|
| `run_tts_test.py` | CustomVoice | Basic TTS with predefined speakers (Ryan, Aiden, etc.) |
| `template_voice_design.py` | VoiceDesign | Create voices from natural language descriptions |
| `template_voice_clone.py` | Base | Clone voices from reference audio |
| `template_voice_design_then_clone.py` | VoiceDesign + Base | Design voice, then clone for consistent generation |
| `template_tokenizer.py` | Tokenizer | Encode/decode audio to/from tokens |

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
