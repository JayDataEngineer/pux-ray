"""Quantization utilities for TRELLIS.2 pipeline.

Drop-in precision changes — no model code modification needed.
Supports: fp32, fp16 (default), bf16, int8 (dynamic), int8 (static calibrate).
"""
from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

log = logging.getLogger("trellis.quantize")

Precision = Literal["fp32", "fp16", "bf16", "int8", "int4"]


def _dtype_for(precision: Precision) -> torch.dtype | None:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": None,   # quantization, not dtype
        "int4": None,
    }[precision]


def quantize_dynamic_int8(module: nn.Module) -> None:
    """Apply PyTorch dynamic int8 quantization to all eligible Linear layers.

    Weights are quantized to int8; activations are quantized dynamically at runtime.
    Zero dependency overhead — uses torch.ao.quantization built-in.
    """
    from torch.ao.quantization import quantize_dynamic

    # Only quantize Linear layers — SparseLinear and SparseConv3d
    # don't support int8 quantization in PyTorch's built-in system.
    quantize_dynamic(
        module,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    log.info("Applied dynamic int8 quantization to Linear layers")


def quantize_dynamic_int4(module: nn.Module) -> None:
    """Apply int4 quantization via bitsandbytes (requires `pip install bitsandbytes`).

    Weights are compressed to 4-bit; forward pass dequantizes on-the-fly.
    Significantly lower VRAM but marginally slower compute.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes required for int4 quantization. Install: pip install bitsandbytes"
        )

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlinear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=True,
                quant_type="nf4",
            )
            qlinear.weight.data = child.weight.data
            if child.bias is not None:
                qlinear.bias.data = child.bias.data
            setattr(module, name, qlinear)
        elif len(list(child.children())) > 0:
            quantize_dynamic_int4(child)

    log.info("Applied int4 quantization (bitsandbytes nf4)")


def apply_precision(pipeline, precision: Precision) -> None:
    """Apply precision setting to a loaded pipeline.

    Called after from_pretrained(), before first inference.

    Args:
        pipeline: Trellis2ImageTo3DPipeline instance
        precision: one of fp32, fp16, bf16, int8, int4
    """
    models = pipeline.models

    if precision in ("fp32", "fp16", "bf16"):
        dtype = _dtype_for(precision)
        for model in models.values():
            model.to(dtype=dtype)
        for extra in ("image_cond_model", "rembg_model"):
            obj = getattr(pipeline, extra, None)
            if isinstance(obj, nn.Module):
                obj.to(dtype=dtype)
        log.info("Pipeline cast to %s", precision)

    elif precision == "int8":
        for model in models.values():
            quantize_dynamic_int8(model)
        for extra in ("image_cond_model", "rembg_model"):
            obj = getattr(pipeline, extra, None)
            if isinstance(obj, nn.Module):
                quantize_dynamic_int8(obj)

    elif precision == "int4":
        for model in models.values():
            quantize_dynamic_int4(model)
        for extra in ("image_cond_model", "rembg_model"):
            obj = getattr(pipeline, extra, None)
            if isinstance(obj, nn.Module):
                quantize_dynamic_int4(obj)

    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / (1024**2)
    log.info("Precision: %s  VRAM=%.0fMB", precision, vram)
