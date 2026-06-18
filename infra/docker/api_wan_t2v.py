"""Wan2.1 T2V 14B API server — text-to-video generation via diffusers.

Uses the Wan2.1-T2V-14B-Diffusers model with sequential CPU offload so the
28 GB BF16 weights fit on a 24 GB RTX 4090 (~1.5 GB resident VRAM at the
cost of ~3x slowdown vs full GPU). For cards with >=40 GB VRAM, set
WAN_OFFLOAD=none for maximum speed.

Although the user's preference is "vLLM-Omni first", the current
vllm-omni:latest build's CLI doesn't expose a working layerwise-offload
flag for BF16 14B diffusion models (the old --enable-layerwise-offload
was removed; --cpu-offload-gb alone OOMs on load). The Boogu-Edit pattern
(custom FastAPI server using diffusers + enable_sequential_cpu_offload)
is the next-best option that still runs in the omni-vllm pool image.

Container image: vllm/vllm-omni:latest (or tech-noir/vllm-omni:fork).
Endpoints:
  GET  /health
  POST /generate  — text → MP4 video
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

MODEL_PATH = os.environ.get(
    "WAN_MODEL_PATH", "/mnt/data/models/video/wan2.1-t2v-14b"
)
DEVICE = os.environ.get("WAN_DEVICE", "cuda:0")
DTYPE = torch.bfloat16
# offload strategies:
#   none       — model fully on GPU (needs ~28 GB VRAM)
#   model_cpu  — diffusers enable_model_cpu_offload (~22 GB VRAM, fastest that fits on 24 GB)
#   sequential — diffusers enable_sequential_cpu_offload (~2 GB VRAM, slowest but always fits)
OFFLOAD = os.environ.get("WAN_OFFLOAD", "sequential")
PORT = int(os.environ.get("WAN_PORT", "8001"))

app = FastAPI(title="Wan2.1 T2V")
_pipe = None
_loaded = False


def load_model():
    global _pipe, _loaded
    if _loaded and _pipe is not None:
        return _pipe

    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Model dir not found: {MODEL_PATH}")

    print(f"Wan-T2V: loading from {MODEL_PATH} (offload={OFFLOAD})")
    t0 = time.perf_counter()

    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import UMT5EncoderModel
    from diffusers.schedulers import UniPCMultistepScheduler

    # Load components individually so we can place them in VRAM strategically.
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=DTYPE
    )
    # Load transformer in BF16; CPU placement first, offload strategy moves it later.
    from diffusers import WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )

    _pipe = WanPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        torch_dtype=DTYPE,
    )

    # Apply offload strategy
    if OFFLOAD == "sequential":
        _pipe.enable_sequential_cpu_offload(device=torch.device(DEVICE))
        print("Wan-T2V: sequential CPU offload enabled (~2 GB VRAM)")
    elif OFFLOAD == "model_cpu":
        _pipe.enable_model_cpu_offload(device=torch.device(DEVICE))
        print("Wan-T2V: model CPU offload enabled (~22 GB VRAM)")
    else:
        _pipe.to(torch.device(DEVICE))
        print("Wan-T2V: no offload (~28 GB VRAM required)")

    # VAE tiling for memory-efficient decode
    _pipe.vae.enable_tiling()

    t1 = time.perf_counter()
    print(f"Wan-T2V: loaded in {t1 - t0:.1f}s")
    _loaded = True
    return _pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the video")
    num_frames: int = Field(default=33, description="Number of frames (8 fps default)")
    height: int = Field(default=480)
    width: int = Field(default=832)  # 480p widescreen
    num_inference_steps: int = Field(default=20)
    guidance_scale: float = Field(default=5.0)
    negative_prompt: str = Field(default="")
    seed: int = Field(default=-1)


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
def generate(req: GenerateRequest):
    if not _loaded:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"model load failed: {e}")

    if req.seed >= 0:
        torch.manual_seed(req.seed)
        import numpy as np
        np.random.seed(req.seed)
        g = torch.Generator(device="cpu").manual_seed(req.seed)
    else:
        g = None

    t0 = time.perf_counter()
    try:
        out = _pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            num_frames=req.num_frames,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=g,
        ).frames[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")

    elapsed = round(time.perf_counter() - t0, 2)

    # Export to MP4 (in-memory)
    import io as _io
    buf = _io.BytesIO()
    import imageio
    # frames: list of HxWxC uint8 numpy arrays
    imageio.mimsave(buf, out, format="MP4", fps=8, codec="libx264")
    mp4_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    gc.collect()

    return Response(
        content=mp4_bytes,
        media_type="video/mp4",
        headers={
            "X-Inference-Time-S": str(elapsed),
            "X-Num-Frames": str(req.num_frames),
            "X-Resolution": f"{req.width}x{req.height}",
            "X-Steps": str(req.num_inference_steps),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
