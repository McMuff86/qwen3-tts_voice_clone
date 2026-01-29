"""
Voice Recorder for Voice Cloning
================================
Record your voice and save it as a template for voice cloning.
Files are saved to: assets/voices/
"""

import os
import shutil
from datetime import datetime
import gradio as gr

# Output directory for voice templates
VOICES_DIR = "assets/voices"
os.makedirs(VOICES_DIR, exist_ok=True)

def process_audio(audio_path, filename):
    if audio_path is None:
        return None, "No audio recorded."
    
    # Generate filename
    if not filename or filename.strip() == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}"
    
    # Ensure .wav extension
    if not filename.endswith(".wav"):
        filename = filename + ".wav"
    
    # Output path
    output_path = os.path.join(VOICES_DIR, filename)
    
    # Copy the recorded file (already WAV from Gradio)
    shutil.copy(audio_path, output_path)
    
    return output_path, f"Saved: {output_path}"

with gr.Blocks(title="Voice Recorder") as demo:
    gr.Markdown("# 🎤 Voice Recorder for Voice Cloning")
    gr.Markdown("Record your voice to use as a template for voice cloning.")
    gr.Markdown("**Tip:** Speak clearly for 3-10 seconds. The recording will be saved to `assets/voices/`")
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], 
            type="filepath", 
            label="🎤 Record your voice"
        )
    
    with gr.Row():
        filename_input = gr.Textbox(
            label="Filename (optional)", 
            placeholder="my_voice (leave empty for auto-generated name)",
            max_lines=1
        )
    
    with gr.Row():
        save_btn = gr.Button("💾 Save Recording", variant="primary")
    
    with gr.Row():
        output_file = gr.File(label="📁 Saved File")
        status_text = gr.Textbox(label="Status", interactive=False)
    
    save_btn.click(
        fn=process_audio,
        inputs=[audio_input, filename_input],
        outputs=[output_file, status_text]
    )

demo.launch()
