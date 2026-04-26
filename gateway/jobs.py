"""Ray Jobs — one-shot @ray.remote tasks for long-running generation.

Creative tools (TRELLIS, AniGen, etc.) and ComfyUI workflows are queued
as Ray remote tasks instead of blocking Ray Serve handlers. Ray's
scheduler handles resource allocation and queuing. Results are stored
in Ray's distributed object store.

Architecture:
  - JobManager (named detached @ray.remote actor): tracks job state
  - @ray.remote functions: run one-shot generation in worker processes
  - HTTP ingress: POST /jobs/<type> submit, GET /jobs/<id> status/result
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import ray

from registry.config import Config

logger = logging.getLogger(__name__)

_TIMEOUT = 600  # 10 min for heavy 3D generation


# ── Helper ────────────────────────────────────────────────────────────────────

def _resolve_config_paths(tool_key: str):
    """Resolve venv_python, script, working_dir from config (relative to project_root)."""
    config = Config()
    root = config.project_root

    def resolve(key: str) -> Path:
        raw = config.get(f"{tool_key}.{key}", "")
        p = Path(raw)
        return p if p.is_absolute() else (root / p).resolve()

    return resolve("venv_python"), resolve("script"), resolve("working_dir")


def _run_tool(venv_python: Path, script: Path, working_dir: Path, args: list[str]) -> bytes:
    """Run a CLI tool via subprocess, return stdout bytes on success."""
    cmd = [str(venv_python), str(script)] + args
    logger.info("Job running: %s", " ".join(cmd[:6]))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        cwd=str(working_dir),
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        stderr = result.stderr[-500:] if result.stderr else b"no stderr"
        raise RuntimeError(f"Tool failed (exit {result.returncode}): {stderr.decode(errors='replace')}")
    return result.stdout


# ── @ray.remote Task Functions ───────────────────────────────────────────────

@ray.remote(num_gpus=0.01)
def run_trellis_job(image: bytes, resolution: int = 512,
                    decimation: int = 1_000_000, seed: int = 0) -> bytes:
    """Generate 3D mesh from image via TRELLIS.2. Returns GLB bytes."""
    venv, script, cwd = _resolve_config_paths("services.creative.trellis")

    with tempfile.TemporaryDirectory(prefix="trellis_job_") as tmpdir:
        input_path = Path(tmpdir) / "input.png"
        output_path = Path(tmpdir) / "output.glb"
        input_path.write_bytes(image)

        _run_tool(venv, script, cwd, [
            "--image", str(input_path),
            "--output", str(output_path),
            "--resolution", str(resolution),
            "--decimation", str(decimation),
        ])

        if not output_path.exists():
            raise RuntimeError(f"TRELLIS did not produce output at {output_path}")
        return output_path.read_bytes()


@ray.remote(num_gpus=0.01)
def run_anigen_job(image: bytes, ss_model: str = "ckpts/anigen/ss_flow_duet",
                   slat_model: str = "ckpts/anigen/slat_flow_auto",
                   seed: int = 42) -> dict[str, bytes]:
    """Generate rigged 3D mesh + skeleton via AniGen."""
    venv, script, cwd = _resolve_config_paths("services.creative.anigen")

    with tempfile.TemporaryDirectory(prefix="anigen_job_") as tmpdir:
        input_path = Path(tmpdir) / "input.png"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        input_path.write_bytes(image)

        _run_tool(venv, script, cwd, [
            "--image_path", str(input_path),
            "--output_dir", str(output_dir),
            "--ss_flow_path", ss_model,
            "--slat_flow_path", slat_model,
            "--seed", str(seed),
        ])

        result = {}
        for glb in sorted(output_dir.rglob("*.glb")):
            if "mesh" in glb.name.lower():
                result["mesh"] = glb.read_bytes()
            if "skeleton" in glb.name.lower():
                result["skeleton"] = glb.read_bytes()
            if "texture" in glb.name.lower():
                result["texture"] = glb.read_bytes()

        if "mesh" not in result:
            raise RuntimeError("AniGen did not produce mesh.glb")
        return result


@ray.remote(num_gpus=0.01)
def run_ace_step_job(
    task_type: str = "text2music",
    prompt: str = "",
    lyrics: str = "",
    audio: bytes | None = None,
    duration: int = 30,
    bpm: int = 120,
    instrumental: bool = True,
    seed: int = -1,
    audio_format: str = "wav",
    # cover params
    cover_strength: float = 1.0,
    # repaint params
    repaint_start: float = 0.0,
    repaint_end: float | None = None,
    repaint_mode: str = "balanced",
    # lego/extract track name
    track_name: str = "",
    # complete tracks
    complete_tracks: str = "",
) -> bytes:
    """Generate or transform music via ACE-Step CLI.

    task_type: text2music | cover | repaint | lego | extract | complete
    - text2music: prompt -> audio
    - cover:      audio + prompt -> remix/style transfer
    - repaint:    audio + prompt + time range -> edited region
    - lego:       audio + track_name -> add instrument layer
    - extract:    audio + track_name -> isolate stem
    - complete:   audio -> auto-extend
    """
    venv, script, cwd = _resolve_config_paths("services.creative.ace_step")

    with tempfile.TemporaryDirectory(prefix="acestep_job_") as tmpdir:
        # Build CLI args
        args = [
            "--task_type", task_type,
            "--save_dir", tmpdir,
            "--seed", str(seed),
            "--batch_size", "1",
            "--audio_format", audio_format,
            "--use_random_seed", str(seed == -1).lower(),
        ]

        if prompt:
            args += ["--caption", prompt]
        if lyrics:
            args += ["--lyrics", lyrics]
        if duration:
            args += ["--duration", str(duration)]
        if bpm:
            args += ["--bpm", str(bpm)]
        args += ["--instrumental", str(instrumental).lower()]

        # Audio input for cover/repaint/lego/extract/complete
        if audio and task_type in ("cover", "repaint", "lego", "extract", "complete"):
            src_path = Path(tmpdir) / "src_audio.wav"
            src_path.write_bytes(audio)
            args += ["--src_audio", str(src_path)]

        # Task-specific params
        if task_type == "cover":
            args += ["--audio_cover_strength", str(cover_strength)]
        elif task_type == "repaint":
            args += ["--repainting_start", str(repaint_start)]
            if repaint_end is not None:
                args += ["--repainting_end", str(repaint_end)]
            args += ["--repaint_mode", repaint_mode]
        elif task_type in ("lego", "extract"):
            if track_name:
                if task_type == "lego":
                    args += ["--lego_track", track_name]
                else:
                    args += ["--extract_track", track_name]
        elif task_type == "complete":
            if complete_tracks:
                args += ["--complete_tracks", complete_tracks]

        _run_tool(venv, script, cwd, args)

        # Find generated audio (newest file with audio extension)
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".opus"}
        for f in sorted(Path(tmpdir).iterdir(),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if f.suffix.lower() in audio_exts and f.name != "src_audio.wav":
                return f.read_bytes()

        raise RuntimeError("ACE-Step did not produce audio output")


# ── JobManager (named detached actor) ─────────────────────────────────────────

@ray.remote(num_gpus=0)
class JobManager:
    """Tracks submitted Ray remote tasks and provides status/result access.

    Deployed as a named detached actor alongside the GPU Scheduler.
    """

    def __init__(self):
        self._jobs: dict[str, tuple[str, ray.ObjectRef]] = {}

    def submit(self, job_type: str, **kwargs) -> str:
        """Submit a job. Returns job_id.

        Supported types: trellis, anigen, ace_step, comfyui.
        """
        job_id = uuid.uuid4().hex[:12]

        if job_type == "trellis":
            ref = run_trellis_job.remote(
                image=kwargs["image"],
                resolution=kwargs.get("resolution", 512),
                decimation=kwargs.get("decimation", 1_000_000),
                seed=kwargs.get("seed", 0),
            )
        elif job_type == "anigen":
            ref = run_anigen_job.remote(
                image=kwargs["image"],
                ss_model=kwargs.get("ss_model", "ckpts/anigen/ss_flow_duet"),
                slat_model=kwargs.get("slat_model", "ckpts/anigen/slat_flow_auto"),
                seed=kwargs.get("seed", 42),
            )
        elif job_type == "ace_step":
            ref = run_ace_step_job.remote(
                task_type=kwargs.get("task_type", "text2music"),
                prompt=kwargs.get("prompt", ""),
                lyrics=kwargs.get("lyrics", ""),
                audio=kwargs.get("audio"),
                duration=kwargs.get("duration", 30),
                bpm=kwargs.get("bpm", 120),
                instrumental=kwargs.get("instrumental", True),
                seed=kwargs.get("seed", -1),
                audio_format=kwargs.get("audio_format", "wav"),
                cover_strength=kwargs.get("cover_strength", 1.0),
                repaint_start=kwargs.get("repaint_start", 0.0),
                repaint_end=kwargs.get("repaint_end"),
                repaint_mode=kwargs.get("repaint_mode", "balanced"),
                track_name=kwargs.get("track_name", ""),
                complete_tracks=kwargs.get("complete_tracks", ""),
            )
        elif job_type == "comfyui":
            ref = run_comfyui_job.remote(
                workflow=kwargs["workflow"],
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

        self._jobs[job_id] = (job_type, ref)
        logger.info("Job submitted: %s (%s)", job_id, job_type)
        return job_id

    def status(self, job_id: str) -> dict:
        """Get job status: {job_id, type, status: running|completed|error|not_found}."""
        entry = self._jobs.get(job_id)
        if entry is None:
            return {"job_id": job_id, "status": "not_found"}

        job_type, ref = entry
        ready, _ = ray.wait([ref], timeout=0)
        if ready:
            try:
                ray.get(ref)
                return {"job_id": job_id, "type": job_type, "status": "completed"}
            except Exception as e:
                return {"job_id": job_id, "type": job_type, "status": "error", "error": str(e)}
        return {"job_id": job_id, "type": job_type, "status": "running"}

    def result(self, job_id: str):
        """Get job result. Returns raw data (bytes or dict). Blocks until complete."""
        entry = self._jobs.get(job_id)
        if entry is None:
            return None

        _job_type, ref = entry
        try:
            return ray.get(ref)
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e)
            raise

    def list_jobs(self) -> list[dict]:
        """List all jobs with status."""
        result = []
        for job_id, (job_type, ref) in self._jobs.items():
            ready, _ = ray.wait([ref], timeout=0)
            status = "completed" if ready else "running"
            result.append({"job_id": job_id, "type": job_type, "status": status})
        return result

    def cleanup(self, job_id: str) -> bool:
        """Remove a completed job from tracking."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


# ── ComfyUI Workflow Job ──────────────────────────────────────────────────────

@ray.remote(num_gpus=0.01)
def run_comfyui_job(workflow: dict) -> dict:
    """Run a ComfyUI workflow as a one-shot job. Returns output info dict.

    Submits the workflow to ComfyUI's /prompt API, polls /history until
    complete, and returns the output file metadata.
    """
    import time

    import httpx

    config = Config()
    root = config.project_root
    raw_port = config.get("services.comfyui.port", 8465)
    comfy_url = f"http://127.0.0.1:{raw_port}"

    # --- Ensure ComfyUI is running ---
    import ray.serve as serve
    try:
        handle = serve.get_deployment_handle("comfyui", "comfyui")
    except Exception:
        raise RuntimeError("ComfyUI deployment not found — is it deployed?")

    with httpx.Client(timeout=30) as cli:
        # Check health
        for _ in range(10):
            try:
                r = cli.get(f"{comfy_url}/", timeout=5)
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(2)
        else:
            raise RuntimeError("ComfyUI not responding after 20s")

        # Submit workflow
        r = cli.post(f"{comfy_url}/prompt", json={"prompt": workflow})
        r.raise_for_status()
        prompt_id = r.json().get("prompt_id")
        logger.info("ComfyUI workflow submitted: %s", prompt_id)

        # Poll for completion
        for _ in range(120):  # 10 min max
            r = cli.get(f"{comfy_url}/history/{prompt_id}")
            r.raise_for_status()
            history = r.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                result = {
                    "prompt_id": prompt_id,
                    "status": history[prompt_id].get("status", {}),
                    "node_outputs": {},
                }
                for node_id, node_output in outputs.items():
                    result["node_outputs"][node_id] = {
                        "images": [
                            {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                             "type": img.get("type", "output")}
                            for img in node_output.get("images", [])
                        ],
                        "gifs": [
                            {"filename": g["filename"], "subfolder": g.get("subfolder", ""),
                             "type": g.get("type", "output")}
                            for g in node_output.get("gifs", [])
                        ],
                    }
                return result
            time.sleep(5)

        raise TimeoutError(f"ComfyUI workflow {prompt_id} timed out")
