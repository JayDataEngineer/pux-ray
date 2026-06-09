"""kohya_ss Forge service — LoRA training via subprocess.

Subprocess-managed GPU. Wraps kohya_ss sd-scripts for LoRA training
on Lance 3B (or Klein 4B fallback).

Supports:
  - Lance 3B LoRA (Qwen2.5-VL backbone, dual-stream MoE)
  - Klein 4B LoRA (Qwen3 text-only backbone, Apache 2.0)
  - Resume from checkpoint
  - VRAM-aware training (AdamW8bit, gradient checkpointing)

Training config: configs/lance_poseedit.toml (in lora project)
Dataset: kohya_ss control_dirs format (img/, Control1-3/, captions/)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

_KOHYA_DIR = Path(os.environ.get("KOHYA_DIR", "/home/user/Documents/programs/lora/.kohya_ss"))
_LORA_PROJECT = Path(os.environ.get("LORA_PROJECT", "/home/user/Documents/programs/lora"))
_TRAIN_CONFIG = _LORA_PROJECT / "configs" / "lance_poseedit.toml"
_TRAIN_SCRIPT = _LORA_PROJECT / "train_lance.sh"
_VENV_PYTHON = _LORA_PROJECT / ".venv" / "bin" / "python"


class KohyaForgeService(ForgeService):
    """kohya_ss LoRA training — subprocess-managed GPU.

    load()  — verifies kohya_ss is installed, pre-flights model paths
    infer() — runs flux_train_network.py with dataset + config
    unload() — cleans up temp files
    """

    vram_mb = 16384  # Reserve 16 GB for training (AdamW8bit + gradient ckpt)
    service_name = "kohya"
    default_model = "lance-poseedit"

    def __init__(self):
        super().__init__()
        self._dataset_dir: Path | None = None
        self._output_dir: Path | None = None
        self._run_id: str = ""

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Pre-flight: verify kohya_ss, model, and dataset are ready."""
        if not _KOHYA_DIR.exists():
            raise FileNotFoundError(
                f"kohya_ss not found at {_KOHYA_DIR}. "
                "Run: cd {_LORA_PROJECT} && task setup:kohya"
            )
        if not (_KOHYA_DIR / "flux_train_network.py").exists():
            raise FileNotFoundError(
                f"flux_train_network.py not found in {_KOHYA_DIR}"
            )

        self.model_name = model_name
        self._loaded = True
        logger.info("kohya: pre-flight OK, model=%s", model_name)

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None
        self._dataset_dir = None
        self._output_dir = None

    def infer(self, payload: dict) -> dict:
        """Run LoRA training.

        Expected payload:
            action: str           — "train" or "status" or "resume"
            dataset_dir: str      — path to kohya_ss control_dirs
            output_dir: str       — path for checkpoints
            config: str           — path to training TOML (optional, default lance_poseedit.toml)
            steps: int            — override max_train_steps (optional)
            rank: int             — override network_dim (optional)
            resume: bool          — resume from latest checkpoint

        Returns:
            {
                "status": "ok",
                "run_id": "...",
                "steps_completed": 12000,
                "checkpoint_path": "...",
                "output": "...",
            }
        """
        action = payload.get("action", "train")

        if action == "status":
            return self._status()

        if action == "resume":
            return self._resume(payload)

        if action == "train":
            return self._train(payload)

        return {"status": "error", "error": f"Unknown action: {action}"}

    def _train(self, payload: dict) -> dict:
        """Launch LoRA training via subprocess."""
        import uuid

        dataset_dir = Path(payload.get("dataset_dir",
                          str(_LORA_PROJECT / "data" / "dataset")))
        output_dir = Path(payload.get("output_dir",
                         str(_LORA_PROJECT / "output" / "checkpoints")))
        config_path = payload.get("config", str(_TRAIN_CONFIG))
        steps = payload.get("steps", 12000)
        rank = payload.get("rank", 128)

        # Validate dataset
        if not (dataset_dir / "img").exists():
            return {"status": "error",
                    "error": f"No img/ directory in {dataset_dir}. "
                             "Run data generation first."}

        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = uuid.uuid4().hex[:12]
        self._dataset_dir = dataset_dir
        self._output_dir = output_dir

        # Generate dataset config
        dataset_config = self._generate_dataset_config(dataset_dir)

        # Build command — use lora project venv for kohya_ss deps
        python_bin = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
        cmd = [
            python_bin,
            str(_KOHYA_DIR / "flux_train_network.py"),
            "--config_file", config_path,
            "--dataset_config", dataset_config,
            "--output_dir", str(output_dir),
            "--output_name", "lance_poseedit",
            "--pretrained_model_name_or_path", str(self._resolve_model()),
        ]

        # Override steps/rank if specified
        if steps:
            cmd.extend(["--max_train_steps", str(steps)])
        if rank:
            cmd.extend(["--network_dim", str(rank)])

        # Check for resume state
        state_files = sorted(output_dir.glob("lance_poseedit-*.state"))
        if state_files and payload.get("resume", True):
            cmd.extend(["--resume", str(state_files[-1])])
            logger.info("kohya: resuming from %s", state_files[-1])

        logger.info("kohya: launching training...")
        logger.info("  cmd: %s", " ".join(str(c) for c in cmd[:8]) + " ...")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_KOHYA_DIR}:{env.get('PYTHONPATH', '')}"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=86400,  # 24 hour timeout
                cwd=str(_LORA_PROJECT),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Training timed out (24h limit)"}

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-2000:]
            return {"status": "error", "error": f"Training failed:\n{err}"}

        # Find latest checkpoint
        checkpoints = sorted(output_dir.glob("lance_poseedit-*.safetensors"))
        latest_ckpt = str(checkpoints[-1]) if checkpoints else ""

        return {
            "status": "ok",
            "run_id": self._run_id,
            "checkpoint_path": latest_ckpt,
            "output_dir": str(output_dir),
            "output": result.stdout[-2000:] if result.stdout else "",
        }

    def _status(self) -> dict:
        """Report training status."""
        if not self._output_dir or not self._output_dir.exists():
            return {"status": "ok", "state": "idle"}

        checkpoints = sorted(self._output_dir.glob("lance_poseedit-*.safetensors"))
        states = sorted(self._output_dir.glob("lance_poseedit-*.state"))

        return {
            "status": "ok",
            "state": "training" if states else "idle",
            "checkpoints": [str(c) for c in checkpoints],
            "latest_state": str(states[-1]) if states else None,
            "dataset_dir": str(self._dataset_dir) if self._dataset_dir else None,
        }

    def _resume(self, payload: dict) -> dict:
        """Resume from latest checkpoint."""
        payload["resume"] = True
        payload["action"] = "train"
        return self._train(payload)

    def _generate_dataset_config(self, dataset_dir: Path) -> str:
        """Generate kohya_ss dataset_config.toml."""
        config_path = dataset_dir / "dataset_config.toml"
        config_path.write_text(f"""\
[general]
resolution = 1024
caption_extension = ".txt"
enable_bucket = false

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{dataset_dir}/img"
  caption_extension = ".txt"
  class_tokens = "LanceEdit"
  num_repeats = 1
""")
        return str(config_path)

    def _resolve_model(self) -> str:
        """Search standard locations for trainable models."""
        from pathlib import Path as P
        for loc in [
            P("/tmp/klein_model"),
            P("/mnt/4tb/Dataset/models"),
            P("/home/user/Documents/programs/ray/infra/repos/ComfyUI/models/unet"),
            P("/home/user/Documents/programs/ray/opt/wan2gp/ckpts"),
        ]:
            if loc.exists():
                for f in loc.rglob("*.safetensors"):
                    n = f.name.lower()
                    if "klein" in n or ("flux" in n and "4b" in n):
                        return str(f)
        raise FileNotFoundError(
            "No model. Pull: hf download DeepBeepMeep/Flux2 "
            "flux-2-klein-4b.safetensors --local-dir /tmp/klein_model"
        )


