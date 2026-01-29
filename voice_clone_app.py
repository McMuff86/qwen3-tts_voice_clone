"""
Voice Clone Gradio App
======================
A web interface for voice cloning using Qwen3-TTS.

Run with: python voice_clone_app.py
Then open: http://127.0.0.1:7861
"""

import os
import numpy as np
import torch
import soundfile as sf
import gradio as gr
from datetime import datetime
from qwen_tts import Qwen3TTSModel

# =============================================================================
# PATHS
# =============================================================================
VOICES_DIR = "assets/voices"
OUTPUT_DIR = "assets/output"

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
print("Loading Qwen3-TTS model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print("✅ Model loaded!")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_voice_files():
    """Get list of voice files in assets/voices/"""
    if not os.path.exists(VOICES_DIR):
        return []
    files = [f for f in os.listdir(VOICES_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
    return sorted(files)


def clone_voice(
    voice_file_dropdown,
    voice_file_upload,
    reference_text,
    text_to_generate,
    language,
    combine_audio,
    pause_seconds
):
    """Main voice cloning function"""
    
    # Determine which voice file to use
    if voice_file_upload is not None:
        ref_audio_path = voice_file_upload
        voice_name = "uploaded"
    elif voice_file_dropdown:
        ref_audio_path = os.path.join(VOICES_DIR, voice_file_dropdown)
        voice_name = voice_file_dropdown.replace(".wav", "")
    else:
        return None, None, "❌ Please select or upload a voice file!"
    
    if not os.path.exists(ref_audio_path):
        return None, None, f"❌ Voice file not found: {ref_audio_path}"
    
    # Parse text (split by newlines)
    texts = [t.strip() for t in text_to_generate.strip().split("\n") if t.strip()]
    
    if not texts:
        return None, None, "❌ Please enter text to generate!"
    
    # Create voice clone prompt
    status_messages = []
    
    if reference_text and reference_text.strip():
        status_messages.append("Using transcript for better quality...")
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=reference_text.strip(),
            x_vector_only_mode=False,
        )
    else:
        status_messages.append("No transcript provided, using x-vector only mode...")
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            x_vector_only_mode=True,
        )
    
    # Generate audio
    all_audio = []
    sample_rate = None
    
    for i, text in enumerate(texts, 1):
        status_messages.append(f"[{i}/{len(texts)}] Generating: {text[:40]}...")
        
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompt,
        )
        
        sample_rate = sr
        all_audio.append(wavs[0])
    
    # Save individual files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for i, audio in enumerate(all_audio, 1):
        output_file = os.path.join(OUTPUT_DIR, f"{voice_name}_{timestamp}_{i}.wav")
        sf.write(output_file, audio, sample_rate)
        saved_files.append(output_file)
    
    status_messages.append(f"✅ Saved {len(saved_files)} files to {OUTPUT_DIR}/")
    
    # Combine if requested
    combined_audio = None
    combined_file = None
    
    if combine_audio and len(all_audio) > 1:
        pause_samples = int(pause_seconds * sample_rate)
        silence = np.zeros(pause_samples, dtype=all_audio[0].dtype)
        
        combined_parts = []
        for i, audio in enumerate(all_audio):
            combined_parts.append(audio)
            if i < len(all_audio) - 1:
                combined_parts.append(silence)
        
        combined_audio = np.concatenate(combined_parts)
        combined_file = os.path.join(OUTPUT_DIR, f"{voice_name}_{timestamp}_combined.wav")
        sf.write(combined_file, combined_audio, sample_rate)
        status_messages.append(f"✅ Combined file saved: {combined_file}")
    
    # Return results
    if combined_audio is not None:
        return (sample_rate, combined_audio), combined_file, "\n".join(status_messages)
    else:
        # Return first audio if only one
        return (sample_rate, all_audio[0]), saved_files[0], "\n".join(status_messages)


def refresh_voice_list():
    """Refresh the voice file dropdown"""
    return gr.Dropdown(choices=get_voice_files(), value=None)


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Voice Clone App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Voice Clone App")
    gr.Markdown("Clone a voice from an audio file and generate new speech.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Select Voice")
            
            voice_dropdown = gr.Dropdown(
                choices=get_voice_files(),
                label="Voice from assets/voices/",
                info="Select a saved voice template"
            )
            refresh_btn = gr.Button("🔄 Refresh List", size="sm")
            
            gr.Markdown("**— OR —**")
            
            voice_upload = gr.Audio(
                label="Upload Voice File",
                type="filepath",
                sources=["upload"]
            )
            
            reference_text = gr.Textbox(
                label="Transcript (optional)",
                placeholder="What is said in the reference audio...",
                info="Improves quality if you know what's spoken",
                lines=2
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 2. Text to Generate")
            
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter text here...\nEach line becomes a separate audio.",
                lines=6,
                value="Hallo, das ist ein Test der Stimmenklonungs-Technologie.\nIch kann alles sagen, was du möchtest."
            )
            
            language = gr.Dropdown(
                choices=["German", "English", "French", "Spanish", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese"],
                value="German",
                label="Language"
            )
            
            with gr.Row():
                combine_check = gr.Checkbox(label="Combine into one file", value=True)
                pause_slider = gr.Slider(0, 2, value=0.5, step=0.1, label="Pause (seconds)")
    
    with gr.Row():
        generate_btn = gr.Button("🎙️ Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            file_output = gr.File(label="Download File")
        
        with gr.Column():
            status_output = gr.Textbox(label="Status", lines=8, interactive=False)
    
    # Event handlers
    refresh_btn.click(fn=refresh_voice_list, outputs=voice_dropdown)
    
    generate_btn.click(
        fn=clone_voice,
        inputs=[
            voice_dropdown,
            voice_upload,
            reference_text,
            text_input,
            language,
            combine_check,
            pause_slider
        ],
        outputs=[audio_output, file_output, status_output]
    )

# =============================================================================
# LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo.launch(server_port=7861)
