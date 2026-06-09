"""GEM-X Forge service — video-based SOMA mesh extraction.

Subprocess-managed GPU. Wraps NVIDIA's GEM-X monocular video pose estimator.
Takes a video file, outputs per-frame SOMA 77-joint poses (rotations + camera).

Per-frame SOMA data is cached to disk so downstream steps can access it
without re-running GEM-X for each training pair.

Architecture:
  Input:  video_path (string or base64)
  Output: cached SOMA mesh directory per video
          (77-joint rotations, camera params, 2D keypoints per frame)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

# GEM-X installation paths (set during setup)
_GEMX_SRC = os.environ.get("GEMX_SRC", "/opt/gemx")
_GEMX_CKPT = os.environ.get("GEMX_CKPT", "/opt/gemx/inputs/pretrained/gem_soma.ckpt")
_GEMX_CACHE = Path(os.environ.get("GEMX_CACHE", "/mnt/4tb/Dataset/gemx_cache"))


class GemxForgeService(ForgeService):
    """GEM-X video pose estimation — outputs SOMA meshes per frame.

    One video → one GEM-X run → SOMA data for all frames.
    Results cached on disk keyed by video path hash.
    """

    vram_mb = 8192  # ~8 GB VRAM for GEM-X
    service_name = "gemx"
    default_model = "gem_soma"

    def __init__(self):
        super().__init__()
        self._ckpt: Path | None = None
        self._cache_dir = _GEMX_CACHE
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Verify GEM-X is installed and checkpoint is available."""
        ckpt = Path(_GEMX_CKPT)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"GEM-X checkpoint not found: {ckpt}. "
                "Run: cd /opt/gemx && git lfs pull"
            )
        src = Path(_GEMX_SRC)
        if not src.is_dir():
            raise FileNotFoundError(
                f"GEM-X source not found: {src}. "
                "Run: git clone --recursive https://github.com/NVlabs/GEM-X.git /opt/gemx"
            )
        self._ckpt = ckpt
        self.model_name = model_name
        self._loaded = True
        logger.info("gemx: loaded ckpt=%s", ckpt)

    def unload(self) -> None:
        self._ckpt = None
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        """Run GEM-X on a video file. Returns cached SOMA data path.

        Expected payload:
            video_path: str        — path to video file
            OR
            video_b64: str         — base64-encoded video (will be decoded to temp)

        Returns:
            {
                "status": "ok",
                "cache_dir": "/mnt/4tb/Dataset/gemx_cache/<video_hash>/",
                "num_frames": 81,
                "soma_format": "rotations_77j_6d",
                "per_frame": {"0000": {"rotations": [...], "camera": {...}}, ...}
            }

        On first run for a video, runs GEM-X and caches. Subsequent calls
        return cached data instantly.
        """
        if not self._loaded or not self._ckpt:
            return {"status": "error", "error": "GEM-X not loaded"}

        video_path = self._resolve_video(payload)
        if not video_path:
            return {"status": "error", "error": "No video_path or video_b64 provided"}

        # Check cache
        video_hash = self._hash_path(video_path)
        cache_dir = self._cache_dir / video_hash

        if cache_dir.exists() and (cache_dir / "metadata.json").exists():
            logger.info("gemx: cache hit for %s", video_path)
            return self._read_cache(cache_dir)

        # Run GEM-X
        try:
            result = self._run_gemx(video_path, cache_dir)
            return result
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "GEM-X timed out (300s limit)"}
        except Exception as e:
            logger.error("gemx: inference failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _resolve_video(self, payload: dict) -> Path | None:
        """Resolve video from payload — path, base64, or temp decode."""
        if "video_path" in payload:
            p = Path(payload["video_path"])
            if p.exists():
                return p

        if "video_b64" in payload:
            tmp = Path(tempfile.gettempdir()) / f"gemx_input_{os.getpid()}.mp4"
            tmp.write_bytes(base64.b64decode(payload["video_b64"]))
            return tmp

        return None

    def _hash_path(self, path: Path) -> str:
        """Short hash of path for cache key."""
        import hashlib
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]

    def _run_gemx(self, video_path: Path, cache_dir: Path) -> dict:
        """Run GEM-X demo script, capture per-frame SOMA data.

        Uses a custom extraction script that runs the GEM pipeline and
        dumps per-frame SOMA rotations + camera params as JSON.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        extract_script = Path(__file__).parent / "extract_soma.py"
        if not extract_script.exists():
            # Fallback: inline the extraction script
            extract_script = self._write_extract_script()

        cmd = [
            sys.executable, str(extract_script),
            "--video", str(video_path),
            "--ckpt", str(self._ckpt),
            "--output", str(cache_dir),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_GEMX_SRC}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = "0"

        logger.info("gemx: running on %s → %s", video_path.name, cache_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-1000:]
            raise RuntimeError(f"GEM-X failed: {err}")

        return self._read_cache(cache_dir)

    def _write_extract_script(self) -> Path:
        """Write the GEM-X extraction script inline."""
        script = Path(__file__).parent / "extract_soma.py"
        script.write_text('''"""GEM-X SOMA extraction — video → per-frame 77-joint 6D rotations + camera.

Usage: python extract_soma.py --video <path> --ckpt <path> --output <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure GEM-X is on path
sys.path.insert(0, "/opt/gemx")


def extract(video_path: str, ckpt_path: str, output_dir: str):
    """Run GEM-X pipeline and save per-frame SOMA data."""
    from gem.pipeline import GEMPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    pipeline = GEMPipeline.from_pretrained(
        ckpt_path,
        device=device,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    # Run on video
    output = pipeline.run_on_video(
        video_path=video_path,
        output_dir=output_dir,
        save_visualizations=False,
    )

    # Extract per-frame SOMA data
    per_frame = {}
    for frame_idx, frame_data in enumerate(output.frames):
        rotations = frame_data.soma_rotations  # (77, 3, 3) or (77, 6)
        if hasattr(rotations, "tolist"):
            rotations = rotations.tolist()
        elif isinstance(rotations, np.ndarray):
            rotations = rotations.tolist()

        camera = {}
        if hasattr(frame_data, "camera"):
            cam = frame_data.camera
            camera = {
                "translation": cam.translation.tolist() if hasattr(cam, "translation") else [0, 0, 0],
                "rotation": cam.rotation.tolist() if hasattr(cam, "rotation") else None,
            }

        per_frame[f"{frame_idx:04d}"] = {
            "rotations": rotations,
            "camera": camera,
        }

    # Write metadata
    meta = {
        "num_frames": len(per_frame),
        "soma_format": "rotations_77j_6d",
        "video_path": video_path,
    }
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(meta, f)

    # Write per-frame data
    with open(Path(output_dir) / "soma_frames.json", "w") as f:
        json.dump(per_frame, f)

    print(json.dumps({"status": "ok", "num_frames": len(per_frame)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract(args.video, args.ckpt, args.output)
''')
        return script

    def _read_cache(self, cache_dir: Path) -> dict:
        """Read cached SOMA data and return standardized response."""
        meta_path = cache_dir / "metadata.json"
        frames_path = cache_dir / "soma_frames.json"

        if not meta_path.exists():
            return {"status": "error", "error": f"Cache incomplete: {cache_dir}"}

        with open(meta_path) as f:
            meta = json.load(f)

        per_frame = {}
        if frames_path.exists():
            with open(frames_path) as f:
                per_frame = json.load(f)

        return {
            "status": "ok",
            "cache_dir": str(cache_dir),
            "num_frames": meta.get("num_frames", 0),
            "soma_format": meta.get("soma_format", "rotations_77j_6d"),
            "per_frame": per_frame,
        }
