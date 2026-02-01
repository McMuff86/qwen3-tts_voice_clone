#!/usr/bin/env python3
"""
CLI Voice Generation Script.
=============================
Supports voice cloning, custom voices, and voice design.

Usage:
    # Voice cloning
    python scripts/clone_from_file.py clone --ref my_voice.wav "Hello world"
    python scripts/clone_from_file.py clone --ref my_voice.wav --lang German "Hallo Welt"
    python scripts/clone_from_file.py clone --ref voice.wav --transcript "ref text" "New text"

    # Custom voice (predefined speakers)
    python scripts/clone_from_file.py custom --speaker Ryan "Hello world"
    python scripts/clone_from_file.py custom --speaker Aiden --instruct "Speak calmly" "Text"

    # Voice design (from description)
    python scripts/clone_from_file.py design --desc "A warm male voice, 40s" "Hello world"

    # Common options (all modes)
    --lang German --model-size 0.6B --combine --pause 0.8 -o output/

    # Default mode is 'clone' for backwards compatibility
    python scripts/clone_from_file.py --ref my_voice.wav "Hello world"
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS voice generation CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common options
    parser.add_argument("--lang", "-l", default=None, help="Target language")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--prefix", "-p", default=None, help="Output filename prefix")
    parser.add_argument("--combine", "-c", action="store_true", help="Combine into one file")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between segments (sec)")
    parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default=None)
    parser.add_argument("--no-save", action="store_true", help="Don't save files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="mode", help="Generation mode")

    # -- clone --
    clone_p = sub.add_parser("clone", help="Clone a voice from reference audio")
    clone_p.add_argument("texts", nargs="+", help="Text(s) to generate")
    clone_p.add_argument("--ref", "-r", required=True, help="Reference audio file")
    clone_p.add_argument("--transcript", "-t", default=None, help="Transcript of reference")

    # -- custom --
    custom_p = sub.add_parser("custom", help="Use a predefined speaker")
    custom_p.add_argument("texts", nargs="+", help="Text(s) to generate")
    custom_p.add_argument("--speaker", "-s", default="Ryan", help="Speaker name")
    custom_p.add_argument("--instruct", "-i", default=None, help="Style instruction")

    # -- design --
    design_p = sub.add_parser("design", help="Design a voice from description")
    design_p.add_argument("texts", nargs="+", help="Text(s) to generate")
    design_p.add_argument("--desc", "-d", required=True, help="Voice description")

    # Backwards compat: allow --ref without subcommand
    parser.add_argument("--ref", "-r", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--transcript", "-t", default=None, help=argparse.SUPPRESS)
    parser.add_argument("texts", nargs="*", default=[], help=argparse.SUPPRESS)

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    # Default mode to 'clone' if --ref is provided without subcommand
    if args.mode is None and args.ref:
        args.mode = "clone"
    elif args.mode is None and args.texts:
        parser.error("Please specify a mode: clone, custom, or design")

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_ref_audio(ref: str) -> Path:
    """Resolve reference audio – check as-is, then in voices dir."""
    path = Path(ref)
    if path.exists():
        return path
    voices_path = config.voices_dir / ref
    if voices_path.exists():
        return voices_path
    print(f"❌ Referenz-Audio nicht gefunden: {ref}")
    print(f"   Gesucht: {path.resolve()}")
    print(f"   Gesucht: {voices_path}")
    sys.exit(1)


def print_progress(i: int, n: int, text: str) -> None:
    """Print progress to stdout."""
    print(f"  [{i}/{n}] {text[:70]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.mode:
        print("Usage: python scripts/clone_from_file.py {clone|custom|design} [options] texts...")
        print("       python scripts/clone_from_file.py --help")
        sys.exit(0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.model_size:
        config.model_size = args.model_size

    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    lang = args.lang or config.default_language

    engine = TTSEngine()

    # ── Clone ──
    if args.mode == "clone":
        ref_path = resolve_ref_audio(args.ref)
        prefix = args.prefix or "clone"

        print(f"🔊 Voice Clone")
        print(f"   Referenz:  {ref_path}")
        print(f"   Sprache:   {lang}")
        print(f"   Modell:    {config.model_size}")
        print(f"   Texte:     {len(args.texts)}")
        print()

        result = engine.clone_voice(
            ref_audio=ref_path,
            texts=args.texts,
            language=lang,
            ref_text=args.transcript,
            output_prefix=prefix,
            save=not args.no_save,
            output_dir=output_dir,
            on_progress=print_progress,
        )

    # ── Custom ──
    elif args.mode == "custom":
        prefix = args.prefix or f"custom_{args.speaker.lower()}"

        print(f"🗣️ Custom Voice")
        print(f"   Sprecher:  {args.speaker}")
        print(f"   Sprache:   {lang}")
        print(f"   Modell:    {config.model_size}")
        print(f"   Texte:     {len(args.texts)}")
        print()

        result = engine.generate_custom(
            texts=args.texts,
            speaker=args.speaker,
            language=lang,
            instruct=args.instruct,
            output_prefix=prefix,
            save=not args.no_save,
            output_dir=output_dir,
            on_progress=print_progress,
        )

    # ── Design ──
    elif args.mode == "design":
        prefix = args.prefix or "designed"

        print(f"✨ Voice Design")
        print(f"   Beschr.:   {args.desc[:60]}...")
        print(f"   Sprache:   {lang}")
        print(f"   Modell:    {config.model_size}")
        print(f"   Texte:     {len(args.texts)}")
        print()

        result = engine.design_voice(
            texts=args.texts,
            voice_description=args.desc,
            language=lang,
            output_prefix=prefix,
            save=not args.no_save,
            output_dir=output_dir,
            on_progress=print_progress,
        )

    else:
        print(f"❌ Unbekannter Modus: {args.mode}")
        sys.exit(1)

    # Combine
    if args.combine and len(result.audio_segments) > 1:
        combined = combine_audio_segments(result.audio_segments, result.sample_rate, args.pause)
        combined_path = output_dir / f"{prefix}_combined.wav"
        save_audio(combined, result.sample_rate, combined_path)
        print(f"\n✅ Kombiniert: {combined_path}")

    print(f"\n🎉 {result.summary}")


if __name__ == "__main__":
    main()
