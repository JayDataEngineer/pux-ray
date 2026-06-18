"""Wan2.1 I2V 14B API server — image+text → video via diffusers.

Uses Wan2.1-I2V-14B-480P-Diffusers with sequential CPU offload so the 28 GB
BF16 weights fit on a 24 GB RTX 4090. The I2V variant adds a CLIP-ViT-H image
encoder (~1.5 GB) on top of the T2V backbone; the encoder runs once at the
start of denoising and the embedding is injected into every transformer block.

Container image: vllm/vllm-omni:latest (or tech-noir/vllm-omni:fork).
Endpoints:
  GET  /health
  POST /generate  — image upload + text prompt → MP4 video

The current vllm-omni CLI does not expose a working layerwise-offload flag
for BF16 14B diffusion models (--cpu-offload-gb alone OOMs on load), so we
use the Boogu-Edit pattern: a custom FastAPI server with diffusers +
enable_sequential_cpu_offload.
"""
from __future__ import annotations

import gc
import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
import uvicorn

MODEL_PATH = os.environ.get(
    "WAN_MODEL_PATH", "/mnt/data/models/video/wan2.1-i2v-14b"
)
DEVICE = os.environ.get("WAN_DEVICE", "cuda:0")
DTYPE = torch.bfloat16
# offload strategies (see api_wan_t2v.py for details):
#   none       — ~30 GB VRAM (14B DiT + CLIP image encoder + VAE)
#   model_cpu  — ~22 GB VRAM, fastest that fits 24 GB
#   sequential — ~2 GB VRAM, slowest but always fits
OFFLOAD = os.environ.get("WAN_OFFLOAD", "sequential")
PORT = int(os.environ.get("WAN_PORT", "8002"))

app = FastAPI(title="Wan2.1 I2V")
_pipe = None
_loaded = False


def load_model():
    global _pipe, _loaded
    if _loaded and _pipe is not None:
        return _pipe

    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Model dir not found: {MODEL_PATH}")

    print(f"Wan-I2V: loading from {MODEL_PATH} (offload={OFFLOAD})")
    t0 = time.perf_counter()

    from diffusers import (
        AutoencoderKLWan,
        WanImageToVideoPipeline,
        WanTransformer3DModel,
    )
    from transformers import (
        UMT5EncoderModel,
        CLIPVisionModel,
    )
    from diffusers.schedulers import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=DTYPE
    )
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    # I2V-specific: CLIP-ViT-H image encoder for the conditioning image.
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_PATH, subfolder="image_encoder", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )

    _pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        scheduler=scheduler,
        torch_dtype=DTYPE,
    )

    if OFFLOAD == "sequential":
        _pipe.enable_sequential_cpu_offload(device=torch.device(DEVICE))
        print("Wan-I2V: sequential CPU offload enabled (~2 GB VRAM)")
    elif OFFLOAD == "model_cpu":
        _pipe.enable_model_cpu_offload(device=torch.device(DEVICE))
        print("Wan-I2V: model CPU offload enabled (~22 GB VRAM)")
    else:
        _pipe.to(torch.device(DEVICE))
        print("Wan-I2V: no offload (~30 GB VRAM required)")

    _pipe.vae.enable_tiling()

    t1 = time.perf_counter()
    print(f"Wan-I2V: loaded in {t1 - t0:.1f}s")
    _loaded = True
    return _pipe


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": _loaded,
        "model_path": MODEL_PATH,
        "offload": OFFLOAD,
    }


@app.post("/load")
def load():
    try:
        load_model()
        return {"status": "ok", "loaded": _loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {e}")


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_frames: int = Form(default=33),
    height: int = Form(default=480),
    width: int = Form(default=832),
    num_inference_steps: int = Form(default=20),
    guidance_scale: float = Form(default=5.0),
    negative_prompt: str = Form(default=""),
    seed: int = Form(default=-1),
):
    """Generate a video from an image + text prompt. Returns MP4 bytes."""
    if not _loaded:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"model load failed: {e}")

    from PIL import Image
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty image upload")

    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    if seed >= 0:
        torch.manual_seed(seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
    else:
        g = None

    t0 = time.perf_counter()
    try:
        out = _pipe(
            image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g,
        ).frames[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")

    elapsed = round(time.perf_counter() - t0, 2)

    buf = io.BytesIO()
    import imageio
    imageio.mimsave(buf, out, format="MP4", fps=8, codec="libx264")
    mp4_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    gc.collect()

    return Response(
        content=mp4_bytes,
        media_type="video/mp4",
        headers={
            "X-Inference-Time-S": str(elapsed),
            "X-Num-Frames": str(num_frames),
            "X-Resolution": f"{width}x{height}",
            "X-Steps": str(num_inference_steps),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
