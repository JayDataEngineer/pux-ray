#!/usr/bin/env python3
"""Boogu-Image-0.1-Edit API server — custom pipeline for image editing & generation.

Boogu-Image-0.1 is a unified image generation and editing model family (Apache-2.0).
This server wraps BooguImagePipeline (a fork of OmniGen2/diffusers) with HTTP
endpoints for T2I (text-to-image) and TI2I (text+image-to-image editing).

Model:   Boogu/Boogu-Image-0.1-Edit  (~35.8 GB)
Arch:    Qwen3-VL MLLM + custom DiT (BooguImageTransformer2DModel) + FLUX.1 VAE
VRAM:    ~40 GB (no offload), ~22 GB (model CPU offload), <2 GB (sequential offload)
         — uses enable_model_cpu_offload by default on 24 GB cards.

Container deps (pre-installed at image build):
  - boogu package (from github.com/boogu-project/Boogu-Image)
  - torch 2.7.1+cu126, diffusers, transformers, flash-attn, cache-dit
"""
from __future__ import annotations

import base64
import gc
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "BOOGU_MODEL_PATH",
    "/mnt/data/models/image-gen/Boogu-Image-0.1-Edit",
)
DEVICE = os.environ.get("BOOGU_DEVICE", "cuda:0")
DTYPE = torch.bfloat16
OFFLOAD = os.environ.get("BOOGU_OFFLOAD", "model_cpu")  # none | model_cpu | sequential
PORT = int(os.environ.get("BOOGU_PORT", "8096"))

# ── Globals ─────────────────────────────────────────────────────────────────
pipe = None
_loaded = False


def load_model():
    global pipe, _loaded
    if _loaded and pipe is not None:
        return

    print(f"Boogu: loading from {MODEL_PATH}")
    print(f"Boogu: device={DEVICE}, offload={OFFLOAD}")
    t0 = time.perf_counter()

    # Import boogu pipeline
    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

    # Load pipeline from pretrained path
    pipe = BooguImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )

    # Apply offload strategy
    if OFFLOAD == "model_cpu":
        pipe.enable_model_cpu_offload(device=torch.device(DEVICE))
        print("Boogu: model CPU offload enabled (~22 GB VRAM)")
    elif OFFLOAD == "sequential":
        pipe.enable_sequential_cpu_offload(device=torch.device(DEVICE))
        print("Boogu: sequential CPU offload enabled (<2 GB VRAM)")
    else:
        pipe.to(torch.device(DEVICE))
        print("Boogu: no offload (~40 GB VRAM required)")

    t1 = time.perf_counter()
    print(f"Boogu: loaded in {t1 - t0:.1f}s")
    _loaded = True


def unload_model():
    global pipe, _loaded
    pipe = None
    _loaded = False
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ── FastAPI app ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(title="Boogu-Image-0.1-Edit", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text instruction for generation")
    negative_prompt: str = Field(
        default="(((deformed))), blurry, over saturation, bad anatomy, "
                "disfigured, poorly drawn face, mutation, mutated, "
                "(extra_limb), (ugly), (poorly drawn hands), fused fingers, "
                "messy drawing, broken legs censor, censored, censor_bar",
        description="Negative prompt",
    )
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    text_guidance_scale: float = 4.0
    image_guidance_scale: float = 1.0
    seed: int = -1
    num_images: int = 1
    # TI2I (image editing) fields
    input_image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded input image for TI2I editing. Omit for T2I.",
    )


class GenerateResponse(BaseModel):
    status: str
    images_b64: list[str] = []
    width: int = 0
    height: int = 0
    time_s: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _loaded, "model": MODEL_PATH}


@app.post("/load")
def load():
    load_model()
    return {"status": "ok"}


@app.post("/release")
def release():
    unload_model()
    return {"status": "ok"}


@app.post("/v1/images/generations")
def generate(req: GenerateRequest):
    if not _loaded or pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare seed
    seed = req.seed if req.seed >= 0 else int(time.time() * 1000) % (2**31)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Prepare input images for TI2I
    input_images = None
    if req.input_image_b64:
        try:
            raw = base64.b64decode(req.input_image_b64)
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            input_images = [[pil_img]]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input image: {e}")

    t0 = time.perf_counter()

    with torch.no_grad():
        results = pipe(
            instruction=req.prompt,
            input_images=input_images,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            text_guidance_scale=req.text_guidance_scale,
            image_guidance_scale=req.image_guidance_scale,
            negative_instruction=req.negative_prompt,
            num_images_per_instruction=req.num_images,
            generator=generator,
            output_type="pil",
            use_rewrite_text_instruction=False,
            cfg_range=(0.0, 1.0),
            use_boosted_orthogonal_guidance=False,
            device=DEVICE,
        )

    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 2)

    # Encode output images
    images_b64 = []
    out_w, out_h = 0, 0
    for img in results.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        images_b64.append(b64)
        if out_w == 0:
            out_w, out_h = img.width, img.height

    return GenerateResponse(
        status="ok",
        images_b64=images_b64,
        width=out_w,
        height=out_h,
        time_s=elapsed,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
