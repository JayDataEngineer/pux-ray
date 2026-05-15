"""Ray job definitions for LoRA voice training on the cluster GPU.

Submits training jobs to the Ray cluster (4090 GPU), handles
preprocessing orchestration, and returns adapter paths.

Usage:
    from qwen_tts.train import train_voice_lora, TrainConfig

    config = TrainConfig(
        character="sakura",
        speaker_name="sakura",
        train_data="/path/to/train_encoded.jsonl",
        val_data="/path/to/val_encoded.jsonl",
    )
    adapter_path = train_voice_lora(config)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "ray://192.168.1.184:10001")


@dataclass
class TrainConfig:
    """Configuration for a LoRA voice training run."""

    character: str
    speaker_name: str
    train_data: str  # path to train_encoded_24k.jsonl
    val_data: str    # path to val_encoded_24k.jsonl
    output_dir: str = ""  # auto-generated if not set

    # Model
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    # LoRA hyperparams
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Training hyperparams
    batch_size: int = 4
    learning_rate: float = 2e-6
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    attn_implementation: str = "flash_attention_2"

    # Runtime
    seed: int = 42
    eval_every: int = 1

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"/tmp/lora-{self.character}"


def _run_lora_training(config: TrainConfig) -> str:
    """Run LoRA training on GPU. This is wrapped as a Ray remote function.

    This function runs INSIDE the Ray cluster, on a GPU node.
    It imports Qwen3-TTS and runs the patched training loop.
    Returns the path to the best checkpoint.
    """
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Qwen3-TTS finetuning scripts are installed in cluster env
    _finetune_root = os.environ.get(
        "QWEN3_TTS_FINETUNE_ROOT",
        "/opt/Qwen3-TTS/finetuning",
    )
    sys.path.insert(0, _finetune_root)

    from sft_12hz_lora import train as _train

    # Build training argv matching sft_12hz_lora.py's argparse
    sys.argv = [
        "sft_12hz_lora.py",
        "--init_model_path", config.base_model,
        "--output_model_path", str(output),
        "--train_jsonl", config.train_data,
        "--val_jsonl", config.val_data,
        "--speaker_name", config.speaker_name,
        "--batch_size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--num_epochs", str(config.num_epochs),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--mixed_precision", config.mixed_precision,
        "--attn_implementation", config.attn_implementation,
        "--lora_rank", str(config.lora_rank),
        "--lora_alpha", str(config.lora_alpha),
        "--lora_dropout", str(config.lora_dropout),
        "--lora_target_modules", config.lora_target_modules,
        "--seed", str(config.seed),
        "--warmup_ratio", str(config.warmup_ratio),
        "--eval_every", str(config.eval_every),
    ]

    _train()

    # Find best checkpoint
    best = output / "best"
    if best.is_dir():
        return str(best)

    checkpoints = sorted(output.glob("checkpoint-epoch-*"))
    if checkpoints:
        return str(checkpoints[-1])

    raise RuntimeError(f"No checkpoints found in {output}")


def train_voice_lora(
    config: TrainConfig,
    address: Optional[str] = None,
    wait: bool = True,
    timeout: float = 7200.0,
) -> str:
    """Submit a LoRA voice training job to the Ray cluster.

    Args:
        config: Training configuration (data paths, hyperparams).
        address: Ray cluster address (default: RAY_ADDRESS env var).
        wait: If True, block until training completes.
        timeout: Max seconds to wait for completion.

    Returns:
        Path to the best LoRA checkpoint on the cluster filesystem.

    Example:
        config = TrainConfig(
            character="sakura",
            speaker_name="sakura",
            train_data="/shared/data/train_encoded_24k.jsonl",
            val_data="/shared/data/val_encoded_24k.jsonl",
        )
        adapter_path = train_voice_lora(config)
        wav = generate("Hello", speaker="sakura", lora_path=adapter_path)
    """
    import ray

    addr = address or RAY_ADDRESS
    if not ray.is_initialized():
        ray.init(address=addr, ignore_reinit_error=True)

    RemoteTrain = ray.remote(num_gpus=1)(_run_lora_training)
    ref = RemoteTrain.remote(config)

    if not wait:
        return str(ref)

    start = time.time()
    while time.time() - start < timeout:
        ready, _ = ray.wait([ref], timeout=5.0)
        if ready:
            return ray.get(ready[0])
        elapsed = time.time() - start
        if int(elapsed) % 30 < 5:
            print(f"  Training... ({elapsed:.0f}s elapsed)", flush=True)

    raise TimeoutError(f"Training timed out after {timeout}s")


def encode_training_data(
    input_jsonl: str,
    output_jsonl: str,
    device: str = "cuda:0",
    address: Optional[str] = None,
) -> str:
    """Encode audio files to codec tokens via Ray cluster GPU.

    Runs Qwen3-TTS's prepare_data.py on the cluster to tokenize
    training audio into the format expected by the LoRA trainer.

    Args:
        input_jsonl: Path to raw training data JSONL.
        output_jsonl: Path for encoded output JSONL.
        device: CUDA device string.
        address: Ray cluster address.

    Returns:
        Path to the encoded JSONL file.
    """
    import ray

    def _encode():
        _finetune_root = os.environ.get(
            "QWEN3_TTS_FINETUNE_ROOT",
            "/opt/Qwen3-TTS/finetuning",
        )
        sys.path.insert(0, _finetune_root)
        from prepare_data import main as _encode_main
        sys.argv = [
            "prepare_data.py",
            "--input_jsonl", input_jsonl,
            "--output_jsonl", output_jsonl,
            "--device", device,
        ]
        _encode_main()
        return output_jsonl

    addr = address or RAY_ADDRESS
    if not ray.is_initialized():
        ray.init(address=addr, ignore_reinit_error=True)

    RemoteEncode = ray.remote(num_gpus=0.5)(_encode)
    ref = RemoteEncode.remote()
    return ray.get(ref)
