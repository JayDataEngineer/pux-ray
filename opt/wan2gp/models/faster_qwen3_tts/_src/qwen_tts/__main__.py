#!/usr/bin/env python3
"""CLI for Qwen3-TTS voice generation via Ray.

Usage:
    uv run -m qwen_tts "Hello world" --speaker Ryan
    uv run -m qwen_tts "Hello" --emotion happy --output hello.wav
    uv run -m qwen_tts "Hello" --lora /path/to/adapter --speaker sakura
    uv run -m qwen_tts --list-speakers
    uv run -m qwen_tts --health
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS voice generation via Ray cluster",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize (omitted with --list-speakers or --health)",
    )
    parser.add_argument(
        "-s", "--speaker",
        default="Ryan",
        help="Speaker name (default: Ryan)",
    )
    parser.add_argument(
        "-e", "--emotion",
        default=None,
        help="Emotion preset name (see --list-emotions)",
    )
    parser.add_argument(
        "-i", "--instruct",
        default=None,
        help="Raw instruct string (overrides --emotion)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output WAV path (default: <speaker>_<emotion>.wav)",
    )
    parser.add_argument(
        "--lora",
        default=None,
        metavar="PATH",
        help="LoRA adapter path on Ray cluster",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.3,
        help="LoRA scale (0.2-0.5, default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (default: auto)",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers and exit",
    )
    parser.add_argument(
        "--list-emotions",
        action="store_true",
        help="List available emotion presets and exit",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check TTS service health and exit",
    )

    args = parser.parse_args()

    # Info commands
    if args.list_emotions:
        from qwen_tts.emotions import EMOTION_PRESETS
        print("Available emotion presets:")
        for name, instruct in sorted(EMOTION_PRESETS.items()):
            instr_preview = instruct[:70] + "..." if len(instruct) > 70 else instruct
            print(f"  {name:15s}  {instr_preview}")
        return

    if args.list_speakers:
        from qwen_tts.client import list_speakers
        print("Available speakers:")
        speakers = list_speakers()
        if speakers:
            for s in speakers:
                print(f"  {s}")
        else:
            print("  (could not reach TTS service)")
        return

    if args.health:
        from qwen_tts.client import health_check
        ok = health_check()
        print("Qwen TTS service: " + ("REACHABLE" if ok else "UNREACHABLE"))
        sys.exit(0 if ok else 1)

    # Generation
    if not args.text:
        parser.error("text is required for generation")

    from qwen_tts.client import generate

    output = args.output
    if output is None:
        tag = args.emotion or args.instruct or "voice"
        output = f"{args.speaker}_{tag}.wav"

    print(f"Generating: '{args.text[:80]}{'...' if len(args.text) > 80 else ''}'")
    print(f"  Speaker: {args.speaker}")
    if args.emotion:
        print(f"  Emotion: {args.emotion}")
    if args.lora:
        print(f"  LoRA: {args.lora} (scale={args.lora_scale})")

    wav_bytes = generate(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        emotion=args.emotion,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        seed=args.seed,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(wav_bytes)

    # Report duration using soundfile
    try:
        with sf.SoundFile(output) as f:
            dur = len(f) / f.samplerate
        print(f"  Saved: {output} ({dur:.1f}s, {len(wav_bytes) / 1024:.0f} KB)")
    except Exception:
        print(f"  Saved: {output} ({len(wav_bytes) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
