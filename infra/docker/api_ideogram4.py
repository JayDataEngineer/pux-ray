#!/usr/bin/env python3
"""Ideogram 4 API server — custom pipeline for vllm-omni container.

Ideogram 4 NF4 uses a fused QKV checkpoint format (3×4608→13824 dim 0)
that diffusers' Ideogram4Pipeline expects as separate to_q/to_k/to_v.

This server loads the NF4 checkpoint, injects attention weights via
the proven dequantize → split → re-quantize approach, and exposes
a T2I endpoint compatible with the forge proxy protocol.

Container: forge-reg.local:30500/tech-noir/vllm-omni:* (has torch, diffusers, bitsandbytes)
Port: 8093
"""
from __future__ import annotations

import gc
import os
import time
from contextlib import asynccontextmanager

import torch
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState
import safetensors
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# ── Model path ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "IDEOGRAM4_MODEL_PATH",
    "/mnt/data/models/image-gen/ideogram4-nf4",
)
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# ── Globals ──────────────────────────────────────────────────────────────────────
pipe = None
_loaded = False

# ── Monkey-patch for meta tensor ─────────────────────────────────────────────────
_orig_apply = torch.nn.Module._apply

def _safe_apply(self, fn):
    try:
        return _orig_apply(self, fn)
    except NotImplementedError as e:
        if "meta tensor" in str(e):
            return self
        raise

torch.nn.Module._apply = _safe_apply


def _inject_attention_weights(comp, comp_name, model_path, device):
    """Dequantize fused QKV/O weights and inject into separate projections."""
    dir_path = os.path.join(model_path, comp_name)
    if not os.path.isdir(dir_path):
        return 0
    loaded = 0
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"):
            continue
        full = os.path.join(dir_path, fname)
        with safetensors.safe_open(full, framework="pt") as f:
            keys = set(f.keys())
        with safetensors.safe_open(full, framework="pt") as f:
            for key in sorted(keys):
                # Fused QKV
                if "attention.qkv.weight" in key and not any(
                    x in key for x in [".absmax", ".quant_map", ".quant_state"]
                ):
                    layer_idx = int(key.split(".")[1])
                    qs = QuantState(
                        absmax=f.get_tensor(key + ".absmax"),
                        code=f.get_tensor(key + ".quant_map"),
                        shape=[13824, 4608],
                        blocksize=64,
                        quant_type="nf4",
                        dtype=torch.float32,
                    )
                    w_float = bnb.functional.dequantize_4bit(f.get_tensor(key), qs)
                    d = w_float.shape[0] // 3
                    attn = comp.layers[layer_idx].attention
                    for pn, wp in [
                        ("to_q", w_float[:d]),
                        ("to_k", w_float[d : 2 * d]),
                        ("to_v", w_float[2 * d :]),
                    ]:
                        proj = getattr(attn, pn)
                        new_p = bnb.nn.Params4bit(wp.contiguous().to(device))
                        proj.weight = new_p
                        new_p._quantize(device)
                        loaded += 1
                    del w_float
                    gc.collect()
                    torch.cuda.empty_cache()
                # Output projection
                elif "attention.o.weight" in key and not any(
                    x in key for x in [".absmax", ".quant_map", ".quant_state"]
                ):
                    layer_idx = int(key.split(".")[1])
                    qs = QuantState(
                        absmax=f.get_tensor(key + ".absmax"),
                        code=f.get_tensor(key + ".quant_map"),
                        shape=[4608, 4608],
                        blocksize=64,
                        quant_type="nf4",
                        dtype=torch.float32,
                    )
                    w_float = bnb.functional.dequantize_4bit(f.get_tensor(key), qs)
                    to_out = comp.layers[layer_idx].attention.to_out[0]
                    new_p = bnb.nn.Params4bit(w_float.contiguous().to(device))
                    to_out.weight = new_p
                    new_p._quantize(device)
                    loaded += 1
                    del w_float
                    gc.collect()
                    torch.cuda.empty_cache()
    return loaded


def load_model():
    global pipe, _loaded
    if _loaded and pipe is not None:
        return

    from diffusers import Ideogram4Pipeline

    print(f"Ideogram4: loading from {MODEL_PATH}")
    t0 = time.perf_counter()

    pipe = Ideogram4Pipeline.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, local_files_only=True
    )

    # Inject attention weights (QKV fused → separate)
    n_t = _inject_attention_weights(pipe.transformer, "transformer", MODEL_PATH, DEVICE)
    n_u = _inject_attention_weights(
        pipe.unconditional_transformer, "unconditional_transformer", MODEL_PATH, DEVICE
    )
    print(f"Injected {n_t + n_u} attention weight(s)")

    pipe.to(DEVICE)

    # Count packed weights
    packed = sum(
        1
        for m in pipe.transformer.modules()
        if isinstance(m, bnb.nn.Linear4bit) and m.weight.device.type == "cuda"
    )
    packed += sum(
        1
        for m in pipe.unconditional_transformer.modules()
        if isinstance(m, bnb.nn.Linear4bit) and m.weight.device.type == "cuda"
    )
    print(f"Packed on CUDA: {packed}")

    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.1f}s")
    _loaded = True


def unload_model():
    global pipe, _loaded
    pipe = None
    _loaded = False
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ── FastAPI app ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Ideogram 4", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 3.0
    max_sequence_length: int = 256
    seed: int = -1

class GenerateResponse(BaseModel):
    status: str
    image_b64: str | None = None
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

    seed = req.seed if req.seed >= 0 else int(time.time() * 1000) % (2**31)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            guidance_schedule=None,
            max_sequence_length=req.max_sequence_length,
            output_type="pil",
            generator=generator,
        )
    t1 = time.perf_counter()

    import io
    import base64

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(
        status="ok",
        image_b64=b64,
        width=result.images[0].width,
        height=result.images[0].height,
        time_s=round(t1 - t0, 2),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8093, log_level="info")
