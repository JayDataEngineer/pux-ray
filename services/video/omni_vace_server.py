"""Omni VACE server — Wan2.2 VACE-Fun via vLLM-Omni.

FastAPI server wrapping vLLM-Omni's Wan22VACEPipeline. Supports two profiles:
  - Base:    FP8, SageAttention, TeaCache 0.025, 18-30 steps  (quality)
  - Turbo:   FP8, LightX2V LoRA, SageAttn2++, Radial Attn, 4 steps  (speed)

API mirrors the existing vace_server.py pattern for forge integration.
Runs on port 8083 (next to DiffSynth's 8082 and sd.cpp's 1234).
"""
from __future__ import annotations

import base64, gc, json, logging, os, threading, time, io
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("omni-vace-server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Omni VACE", version="0.1.0")

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("OMNI_MODEL", "alibaba-pai/Wan2.2-VACE-Fun-A14B")
MODELS_ROOT = os.environ.get("VACE_MODELS_ROOT", "/mnt/data/models/video")
PORT = int(os.environ.get("OMNI_PORT", "8083"))

# Profile defaults
DEFAULT_STEPS = int(os.environ.get("OMNI_STEPS", "18"))
DEFAULT_CFG = float(os.environ.get("OMNI_CFG", "5.0"))

# ── Omni engine singleton ─────────────────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()


def _ensure_engine():
    """Lazy-init Omni engine (takes ~60s to load model)."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        logger.info("Loading Omni engine with model: %s", MODEL_ID)
        t0 = time.perf_counter()
        from vllm_omni.entrypoints.omni import Omni
        _engine = Omni(model=MODEL_ID)
        elapsed = time.perf_counter() - t0
        logger.info("Omni engine loaded in %.1fs", elapsed)
        return _engine


# ── Request / Response schemas ────────────────────────────────────────────

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 81
    steps: int = Field(default=DEFAULT_STEPS, alias="sampling_steps", ge=1, le=50)
    cfg: float = Field(default=DEFAULT_CFG, alias="guide_scale", ge=1.0, le=20.0)
    seed: int = -1
    fps: int = 16
    profile: str = "base"  # "base" or "turbo"
    lora_path: str = ""    # LightX2V lora path for turbo mode
    lora_scale: float = 0.6

    # VACE-specific inputs
    reference_image: Optional[str] = None  # base64 or URL
    source_video: Optional[str] = None     # base64 or URL
    source_mask: Optional[str] = None      # base64 or URL
    last_image: Optional[str] = None       # for FLF2V

    class Config:
        populate_by_name = True


class GenerationResponse(BaseModel):
    status: str
    output: dict = {}
    metrics: dict = {}


# ── Generation ────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest, background: BackgroundTasks):
    omni = _ensure_engine()

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Profile setup
    profile = req.profile.lower()
    steps = req.steps
    if profile == "turbo":
        steps = min(steps, 4)  # LightX2V = max 4 steps
        if req.lora_path:
            logger.info("Turbo mode with LoRA: %s (scale=%.2f)", req.lora_path, req.lora_scale)

    # Build sampling params
    sampling_params = OmniDiffusionSamplingParams(
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        num_inference_steps=steps,
        guidance_scale=req.cfg,
        seed=req.seed if req.seed >= 0 else None,
        fps=req.fps,
    )

    # Build multimodal data
    images = None
    last_images = None
    video = None
    mask = None

    if req.reference_image:
        from PIL import Image
        if req.reference_image.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(req.reference_image) as r:
                images = [Image.open(io.BytesIO(r.read())).convert("RGB")]
        else:
            images = [Image.open(io.BytesIO(base64.b64decode(req.reference_image))).convert("RGB")]

    if req.last_image:
        from PIL import Image
        if req.last_image.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(req.last_image) as r:
                last_images = [Image.open(io.BytesIO(r.read())).convert("RGB")]

    # FLF2V mode: pass both images as first+last keyframes
    all_images = images or []
    if last_images:
        all_images.extend(last_images)
    if not all_images:
        all_images = None

    # Build text prompt
    texts = [req.prompt]
    if req.negative_prompt:
        texts.append(req.negative_prompt)

    t0 = time.perf_counter()
    logger.info("Omni generate: profile=%s prompt=%r steps=%d frames=%d",
                profile, req.prompt[:50], steps, req.num_frames)

    outputs = omni.generate(
        texts,
        sampling_params,
        images=all_images,
        use_async=False,
    )

    elapsed = time.perf_counter() - t0
    video_tensor = outputs[0].images

    # Convert tensor → mp4 bytes
    mp4_bytes = _tensor_to_mp4(video_tensor, fps=req.fps)
    video_b64 = base64.b64encode(mp4_bytes).decode()

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

    return GenerationResponse(
        status="success",
        output={
            "type": "video",
            "content": video_b64,
            "format": "mp4",
            "fps": req.fps,
        },
        metrics={
            "latency_s": round(elapsed, 2),
            "vram_peak_mb": int(peak_vram),
            "steps": steps,
            "num_frames": req.num_frames,
            "profile": profile,
            "model": MODEL_ID,
        },
    )


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if _engine is not None else "idle",
        "model": MODEL_ID,
        "profiles": ["base", "turbo"],
        "engine_loaded": _engine is not None,
    }


@app.post("/load")
async def load():
    _ensure_engine()
    return {"status": "loaded", "model": MODEL_ID}


@app.post("/release")
async def release():
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "released"}


# ── Helper ────────────────────────────────────────────────────────────────

def _tensor_to_mp4(frames, fps=16, quality=5):
    """Convert a list of PIL Image frames to MP4 bytes."""
    from PIL import Image
    import tempfile
    import subprocess

    # Save frames as temp PNGs, encode via ffmpeg
    tmpdir = tempfile.mkdtemp()
    try:
        for i, frame in enumerate(frames):
            if isinstance(frame, torch.Tensor):
                arr = ((frame * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                frame = Image.fromarray(arr)
            frame.save(f"{tmpdir}/frame_{i:04d}.png")
        out_path = f"{tmpdir}/output.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmpdir}/frame_%04d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(quality),
            out_path
        ], capture_output=True, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    logger.info("Omni VACE server starting on port %d", PORT)
    logger.info("Model: %s", MODEL_ID)
    logger.info("Profiles: base (%d steps), turbo (4 steps)", DEFAULT_STEPS)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
