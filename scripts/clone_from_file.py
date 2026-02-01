#!/usr/bin/env python3
"""
CLI Voice Cloning Script.
=========================
Clone a voice from an audio file and generate new speech.

Usage:
    python scripts/clone_from_file.py --ref assets/voices/my_voice.wav "Hello world"
    python scripts/clone_from_file.py --ref my_voice.wav --lang German "Hallo Welt"
    python scripts/clone_from_file.py --ref my_voice.wav --transcript "original text" "New text"
    python scripts/clone_from_file.py --ref my_voice.wav -o output/ "Line 1" "Line 2" "Line 3"

Multiple texts are generated as separate files. Use --combine to merge them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import config
from src.engine import TTSEngine
from src.audio_utils import combine_audio_segments, save_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a voice and generate speech with Qwen3-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ref assets/voices/my_voice.wav "Hello world"
  %(prog)s --ref my_voice.wav --lang German "Hallo Welt" "Wie geht es dir?"
  %(prog)s --ref my_voice.wav --transcript "what is said in audio" "New text"
  %(prog)s --ref my_voice.wav --combine --pause 0.8 "Line 1" "Line 2"
        """,
    )

    parser.add_argument(
        "texts",
        nargs="+",
        help="Text(s) to generate. Each argument becomes a separate audio file.",
    )
    parser.add_argument(
        "--ref", "-r",
        required=True,
        help="Reference audio file (path or filename in assets/voices/).",
    )
    parser.add_argument(
        "--transcript", "-t",
        default=None,
        help="Transcript of the reference audio (improves quality).",
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        help=f"Target language (default: {config.default_language}).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"Output directory (default: {config.output_dir}).",
    )
    parser.add_argument(
        "--prefix", "-p",
        default="clone",
        help="Output filename prefix (default: clone).",
    )
    parser.add_argument(
        "--combine", "-c",
        action="store_true",
        help="Also save a combined file with all texts merged.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause between sentences in combined file, in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--model-size",
        choices=["0.6B", "1.7B"],
        default=None,
        help="Model size to use (default: from config).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save files, just generate (for testing).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def resolve_ref_audio(ref: str) -> Path:
    """Resolve reference audio path – check as-is first, then in voices dir."""
    path = Path(ref)
    if path.exists():
        return path

    # Try in voices directory
    voices_path = config.voices_dir / ref
    if voices_path.exists():
        return voices_path

    print(f"❌ Referenz-Audio nicht gefunden: {ref}")
    print(f"   Gesucht in: {path.resolve()}")
    print(f"   Gesucht in: {voices_path}")
    sys.exit(1)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Override model size if specified
    if args.model_size:
        config.model_size = args.model_size

    ref_path = resolve_ref_audio(args.ref)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir

    print(f"📎 Referenz: {ref_path}")
    print(f"📁 Output:   {output_dir}")
    print(f"🌐 Sprache:  {args.lang or config.default_language}")
    print(f"📝 Texte:    {len(args.texts)}")
    print()

    engine = TTSEngine()

    result = engine.clone_voice(
        ref_audio=ref_path,
        texts=args.texts,
        language=args.lang,
        ref_text=args.transcript,
        output_prefix=args.prefix,
        save=not args.no_save,
        output_dir=output_dir,
    )

    # Combine if requested
    if args.combine and len(result.audio_segments) > 1:
        combined = combine_audio_segments(
            result.audio_segments,
            result.sample_rate,
            args.pause,
        )
        combined_path = output_dir / f"{args.prefix}_combined.wav"
        save_audio(combined, result.sample_rate, combined_path)
        print(f"\n✅ Kombiniert: {combined_path}")

    print(f"\n🎉 Fertig! {len(result.audio_segments)} Audio(s) generiert.")


if __name__ == "__main__":
    main()
