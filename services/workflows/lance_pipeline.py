"""Lance Pose-Edit data pipeline — workflow functions.

Each function is callable from Forge DAG steps (type: python).
No GPU code — all heavy lifting delegated to Forge services.

Pipeline stages:
  1. extract_frames    — ffmpeg video → frame images
  2. gemx_mesh         — GEM-X video → per-frame SOMA meshes (GPU, cached)
  3. dwpose_skeleton   — DWPose frame → skeleton overlay (GPU)
  4. pair_frames       — select Frame_A/Frame_B pairs (CPU)
  5. build_dataset     — assemble kohya_ss control_dirs structure (CPU)
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKFLOWS = [
    {"id": "lance/extract-frames",
     "description": "Extract frames from video via ffmpeg"},
    {"id": "lance/gemx-mesh",
     "description": "GEM-X video-based SOMA mesh extraction (GPU)"},
    {"id": "lance/pair-frames",
     "description": "Pair frames for training (Frame_A ≠ Frame_B)"},
    {"id": "lance/build-dataset",
     "description": "Assemble kohya_ss control_dirs from frame pairs"},
    {"id": "lance/full-pipeline",
     "description": "Complete training data generation from videos"},
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


# ── Stage 1: Extract frames ──────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 81,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract frames from a single video file using ffmpeg.

    Args:
        video_path: Path to .mp4 video file.
        output_dir: Directory to write frame PNGs.
        max_frames: Maximum number of frames to extract.

    Returns:
        {"status": "ok", "output_dir": ..., "num_frames": N, "frames": [...]}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-frames:v", str(max_frames),
        "-q:v", "2",
        f"{out}/{stem}_%04d.png",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        return {"status": "error", "error": result.stderr[-500:]}

    frames = sorted(out.glob(f"{stem}_*.png"))
    return {
        "status": "ok",
        "output_dir": str(out),
        "num_frames": len(frames),
        "frames": [str(f) for f in frames],
    }


# ── Stage 3: Frame pairing ───────────────────────────────────────

def pair_frames(
    frames_dir: str,
    output_json: str = "",
    frame_offset: int = 30,
    pairs_per_video: int = 1,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate Frame_A/Frame_B pairs from extracted frames.

    For each video directory, picks 2 frames at least `frame_offset` apart.
    Frame_A = character reference (Control2).
    Frame_B = target image (img/) + pose source.

    The offset prevents the model from learning identity mapping.
    Same person, genuinely different pose.

    Args:
        frames_dir: Directory with frame subdirectories.
        output_json: If set, write pairs to this JSON file.
        frame_offset: Minimum frame gap between pairs.
        pairs_per_video: Max pairs per video.
        seed: Random seed.

    Returns:
        {"status": "ok", "pairs": [...], "num_pairs": N}
    """
    rng = random.Random(seed)
    base = Path(frames_dir)

    # Support both flat frame files and subdirectory structure
    frame_dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not frame_dirs:
        # Flat structure — treat all frame files as one "video"
        frames = sorted(
            list(base.glob("*.png")) + list(base.glob("*.jpg"))
        )
        if len(frames) < frame_offset + 2:
            return {"status": "ok", "pairs": [], "num_pairs": 0}

        pairs = []
        total = len(frames)
        for _ in range(pairs_per_video):
            max_a = total - frame_offset - 1
            if max_a < 1:
                break
            idx_a = rng.randint(0, max_a)
            idx_b = rng.randint(idx_a + frame_offset, total - 1)
            pairs.append({
                "video": base.name,
                "frame_a_path": str(frames[idx_a]),
                "frame_b_path": str(frames[idx_b]),
                "frame_a_idx": idx_a,
                "frame_b_idx": idx_b,
            })
        return {"status": "ok", "pairs": pairs, "num_pairs": len(pairs)}

    # Directory-per-video structure
    all_pairs = []
    for frame_dir in frame_dirs:
        frames = sorted(
            list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg"))
        )
        if len(frames) < frame_offset + 2:
            continue

        total = len(frames)
        n = min(pairs_per_video, total // max(1, frame_offset))

        for _ in range(n):
            max_a = total - frame_offset - 1
            if max_a < 1:
                break
            idx_a = rng.randint(0, max_a)
            idx_b = rng.randint(idx_a + frame_offset, total - 1)
            all_pairs.append({
                "video": frame_dir.name,
                "frame_a_path": str(frames[idx_a]),
                "frame_b_path": str(frames[idx_b]),
                "frame_a_idx": idx_a,
                "frame_b_idx": idx_b,
            })

    rng.shuffle(all_pairs)

    if output_json:
        Path(output_json).write_text(json.dumps(all_pairs, indent=2))

    return {"status": "ok", "pairs": all_pairs, "num_pairs": len(all_pairs)}


# ── Stage 4: Dataset assembly ─────────────────────────────────────

def build_dataset(
    pairs_json: str,
    output_dir: str,
    soma_cache_dir: str = "",
    resolution: int = 1024,
    **kwargs: Any,
) -> dict[str, Any]:
    """Assemble kohya_ss control_dirs dataset from frame pairs.

    Reads pre-computed frame pairs and SOMA/GEM-X mesh cache,
    assembles the final training dataset structure:

        output_dir/
        ├── img/          # Target frame (Frame_B)
        ├── Control1/     # SOMA mesh render (from GEM-X cache)
        ├── Control2/     # Character reference (Frame_A)
        ├── Control3/     # DWPose skeleton (from Forge step)
        └── captions/     # Text captions

    NO TEACHER MODEL. Target = raw video frame.

    Args:
        pairs_json: Path to JSON file with frame pairs.
        output_dir: Base output directory for kohya_ss structure.
        soma_cache_dir: GEM-X cache directory with per-frame SOMA data.
        resolution: Output image resolution (square).

    Returns:
        {"status": "ok", "num_pairs": N, "output_dir": "..."}
    """
    from PIL import Image

    pairs_path = Path(pairs_json)
    if not pairs_path.exists():
        return {"status": "error", "error": f"Pairs file not found: {pairs_json}"}

    pairs = json.loads(pairs_path.read_text())
    out = Path(output_dir)

    for subdir in ("img", "Control1", "Control2", "Control3", "captions"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    # Load SOMA cache if available
    soma_data = {}
    if soma_cache_dir:
        soma_path = Path(soma_cache_dir) / "soma_frames.json"
        if soma_path.exists():
            soma_data = json.loads(soma_path.read_text())

    generated = 0
    for pair_idx, pair in enumerate(pairs):
        sample_id = f"{pair_idx:06d}"

        try:
            frame_a = Image.open(pair["frame_a_path"]).convert("RGB")
            frame_b = Image.open(pair["frame_b_path"]).convert("RGB")
        except Exception:
            continue

        frame_a = frame_a.resize((resolution, resolution), Image.LANCZOS)
        frame_b = frame_b.resize((resolution, resolution), Image.LANCZOS)

        # img/ = Frame_B (training target — raw video frame)
        frame_b.save(out / "img" / f"{sample_id}.png")

        # Control2 = Frame_A (character reference — different frame, same person)
        frame_a.save(out / "Control2" / f"{sample_id}.png")

        # Control1 = SOMA mesh from GEM-X cache (or blank if unavailable)
        frame_key = f"{pair['frame_b_idx']:04d}"
        if frame_key in soma_data:
            # TODO: render SOMA mesh from rotations to image
            # For now: save blank placeholder
            blank = Image.new("RGB", (resolution, resolution), (255, 255, 255))
            blank.save(out / "Control1" / f"{sample_id}.png")
        else:
            blank = Image.new("RGB", (resolution, resolution), (255, 255, 255))
            blank.save(out / "Control1" / f"{sample_id}.png")

        # Control3 = skeleton placeholder (filled by DWPose Forge step)
        blank = Image.new("RGB", (resolution, resolution), (255, 255, 255))
        blank.save(out / "Control3" / f"{sample_id}.png")

        # Caption
        scene = pair.get("video", "scene")
        caption = (
            f"LanceEdit, character reference from earlier frame, "
            f"scene {scene}, target pose frame {pair.get('frame_b_idx', 0)}"
        )
        (out / "captions" / f"{sample_id}.txt").write_text(caption)

        generated += 1

    return {
        "status": "ok",
        "num_pairs": generated,
        "output_dir": str(out),
    }
