"""Qwen3-TTS voice generation and LoRA training, via Ray Serve.

Inference:
    from qwen_tts import generate, generate_lora
    from qwen_tts.emotions import EMOTION_PRESETS

    wav_bytes = generate("Hello world", speaker="Ryan")
    wav_bytes = generate("Hello world", speaker="Ryan", emotion="happy")
    wav_bytes = generate_lora("Hello", speaker="sakura", lora_path="/path/to/adapter")

Training:
    from qwen_tts import train_voice_lora, TrainConfig

    config = TrainConfig(
        character="sakura",
        speaker_name="sakura",
        train_data="/shared/train_encoded.jsonl",
        val_data="/shared/val_encoded.jsonl",
    )
    adapter_path = train_voice_lora(config)
"""

from qwen_tts.client import (
    generate,
    generate_lora,
    generate_stream,
    list_speakers,
    health_check,
    RAY_API_URL,
    TTS_ENDPOINT,
)
from qwen_tts.emotions import EMOTION_PRESETS, emotion_to_instruct
from qwen_tts.train import train_voice_lora, TrainConfig, encode_training_data

__all__ = [
    "generate",
    "generate_lora",
    "generate_stream",
    "list_speakers",
    "health_check",
    "EMOTION_PRESETS",
    "emotion_to_instruct",
    "train_voice_lora",
    "TrainConfig",
    "encode_training_data",
    "RAY_API_URL",
    "TTS_ENDPOINT",
]
