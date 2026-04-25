#!/usr/bin/env python3
"""
Qwen3-TTS CLI Wrapper
=====================
Simple command-line wrapper for Qwen3-TTS base-model inference.
Use this for one-off voice generation without a trained LoRA adapter.

Usage:
    python qwen3_tts_cli.py --text "Hello world" --output hello.wav
    python qwen3_tts_cli.py --text "Run!" --output run.wav --speaker "sakura" --emotion angry
"""
import argparse
import os
import sys

# Resolve Qwen3-TTS project root (supports env var or cwd)
QWEN_TTS_ROOT = os.environ.get("QWEN_TTS_ROOT", os.getcwd())
QWEN_TTS_PACKAGE = os.path.join(QWEN_TTS_ROOT, "Qwen3-TTS")
sys.path.insert(0, QWEN_TTS_PACKAGE)
sys.path.insert(0, os.path.join(QWEN_TTS_PACKAGE, "finetuning"))

# Ensure CUDA runtime is found (matching vn_generate.py pattern)
cuda_rt = os.path.join(QWEN_TTS_ROOT, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cuda_runtime", "lib")
if os.path.isdir(cuda_rt):
    os.environ["LD_LIBRARY_PATH"] = cuda_rt + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import soundfile as sf
import torch


EMOTION_PRESETS = {
    "happy": "Speak in a cheerful and happy tone, voice light and bright",
    "sad": "Speak sadly, voice slightly trembling, slower pace",
    "angry": "Speak with suppressed anger, voice low, suddenly raising at the end",
    "scared": "Speak with fear and trembling, getting faster as you go",
    "gentle": "Speak gently and softly, as if comforting someone",
    "tsundere": "Speak in a tsundere way, saying harsh things but unable to hide the embarrassment",
    "whisper": "Whisper softly, like telling a secret, breathy voice",
    "cold": "Speak in a cold and distant tone, no emotional fluctuation",
    "excited": "Speak with excitement and enthusiasm, energetic and upbeat",
    "shy": "Speak shyly, hesitant, soft voice with occasional pauses",
    "serious": "Speak seriously and firmly, clear and deliberate",
    "playful": "Speak playfully and teasingly, with a mischievous tone",
    "worried": "Speak with worry and concern, slightly anxious",
    "confident": "Speak with confidence and assurance, strong and clear",
    "tired": "Speak tiredly, slower pace, slightly quieter",
    "surprised": "Speak with genuine surprise, voice raised slightly",
}


def trim_trailing_silence(wav, sr, threshold=0.01, min_silence=0.15):
    min_samples = int(min_silence * sr)
    for i in range(len(wav) - 1, min_samples, -1):
        if abs(wav[i]) > threshold:
            return wav[: i + 1]
    return wav


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS base model inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", required=True, help="Output audio path (.wav)")
    parser.add_argument("--speaker", default="narrator", help="Speaker name")
    parser.add_argument("--language", default="auto", help="Language code")
    parser.add_argument("--emotion", default=None, help=f"Emotion preset: {', '.join(EMOTION_PRESETS.keys())}")
    parser.add_argument("--instruct", default=None, help="Raw instruct prompt (overrides --emotion)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Base model ID")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    instruct = args.instruct
    if instruct is None and args.emotion:
        instruct = EMOTION_PRESETS.get(args.emotion.lower())
        if instruct is None:
            print(f"Warning: unknown emotion '{args.emotion}', using as raw instruct", file=sys.stderr)
            instruct = f"Speak in a {args.emotion} tone"

    print(f"Loading Qwen3-TTS model ({args.model})...")
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        print(f"Error: Cannot import qwen_tts: {e}", file=sys.stderr)
        print(f"Hint: QWEN_TTS_PACKAGE = {QWEN_TTS_PACKAGE}", file=sys.stderr)
        sys.exit(1)

    # Pick attention implementation
    try:
        import flash_attn
        attn = "flash_attention_2"
    except (ImportError, OSError):
        attn = "sdpa"
    print(f"Using attention: {attn}")

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": 1.05,
    }

    print(f"Generating: '{args.text[:60]}{'...' if len(args.text) > 60 else ''}'")
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=instruct,
        **gen_kwargs,
    )

    audio = trim_trailing_silence(wavs[0], sr)
    # Cap runaway generation
    max_samples = int(15.0 * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sf.write(args.output, audio, sr)
    dur = len(audio) / sr
    print(f"Saved: {args.output} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
