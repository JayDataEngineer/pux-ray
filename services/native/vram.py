"""VRAM optimization layer — adaptive format selection and offload.

Implements the component-type rules from the tier architecture:
  Rule 1: VAE is always BF16 (too small to quantize, precision-critical for decode)
  Rule 2: Text encoder is quantizable (invisible quality impact)
  Rule 3: Transformer gets best format VRAM allows (precision-critical)
  Rule 4: group_offload when something doesn't fit (stream blocks via CUDA streams)

Degradation chain:
  BF16 resident → BF16 + group_offload → FP8 resident → FP8 + group_offload → GGUF
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class Format(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"          # flat FP8 via layerwise_casting
    INT8 = "int8"        # optimum-quanto int8
    GGUF_Q8 = "gguf_q8"
    GGUF_Q5 = "gguf_q5"
    GGUF_Q4 = "gguf_q4"


class OffloadStrategy(str, Enum):
    RESIDENT = "resident"            # pipe.to("cuda") — everything in VRAM
    MODEL_CPU_OFFLOAD = "model_cpu"  # enable_model_cpu_offload — component-level swap
    GROUP_OFFLOAD = "group_offload"  # enable_group_offload — block-level streaming
    SEQUENTIAL = "sequential"        # enable_sequential_cpu_offload — layer-by-layer


# ─── VRAM Detection ────────────────────────────────────────────────────────────

def get_available_vram_mb() -> int:
    """Get available VRAM in MB (85% of free, leaving headroom for activations)."""
    try:
        free_bytes, _total = torch.cuda.mem_get_info(0)
        free_mb = int(free_bytes / (1024 * 1024))
        return int(free_mb * 0.85)  # 15% headroom
    except Exception:
        return 0


def get_total_vram_mb() -> int:
    """Get total VRAM in MB."""
    try:
        _free, total_bytes = torch.cuda.mem_get_info(0)
        return int(total_bytes / (1024 * 1024))
    except Exception:
        return 24576  # Default for RTX 4090


# ─── Adaptive Configuration ────────────────────────────────────────────────────

    """The VRAM optimization plan decided by the adaptive loader."""
    strategy: OffloadStrategy
    transformer_format: Format
    text_encoder_format: Format
    vae_format: Format  # Always BF16
    use_compile: bool
    use_cache_accel: bool
    use_group_offload: bool
    estimated_vram_mb: int
    notes: str = ""


# Fix: use dataclass
from dataclasses import dataclass

@dataclass
class VRAMPlan:
    """The VRAM optimization plan decided by the adaptive loader."""
    strategy: OffloadStrategy
    transformer_format: Format
    text_encoder_format: Format
    vae_format: Format
    use_compile: bool
    use_cache_accel: bool
    use_group_offload: bool
    estimated_vram_mb: int
    notes: str = ""


def plan_vram(
    available_mb: int,
    model_bf16_size_mb: int,
    text_encoder_bf16_size_mb: int = 0,
    vae_size_mb: int = 300,
    quality_tier: bool = True,
) -> VRAMPlan:
    """Decide the optimal VRAM configuration based on available memory.

    Args:
        available_mb: Available VRAM in MB
        model_bf16_size_mb: Transformer/DiT size in BF16
        text_encoder_bf16_size_mb: Text encoder size in BF16 (may be 0 if unknown)
        vae_size_mb: VAE size (always kept in BF16)
        quality_tier: True for Quality tier (prefer BF16), False for Speed (prefer FP8)

    Returns:
        VRAMPlan with the selected strategy and formats
    """
    total_vram = get_total_vram_mb()
    activation_headroom = 4096  # Reserve 4GB for activations

    # Estimate component sizes at different formats
    te_fp8 = text_encoder_bf16_size_mb // 2 if text_encoder_bf16_size_mb else 0
    te_int8 = text_encoder_bf16_size_mb // 4 if text_encoder_bf16_size_mb else 0
    model_fp8 = model_bf16_size_mb // 2
    model_int8 = model_bf16_size_mb // 4

    # ── Try BF16 resident first (fastest + best quality) ──────────────────────
    total_bf16 = model_bf16_size_mb + text_encoder_bf16_size_mb + vae_size_mb + activation_headroom
    if total_bf16 <= available_mb:
        return VRAMPlan(
            strategy=OffloadStrategy.RESIDENT,
            transformer_format=Format.BF16,
            text_encoder_format=Format.BF16,
            vae_format=Format.BF16,
            use_compile=True,
            use_cache_accel=True,
            use_group_offload=False,
            estimated_vram_mb=total_bf16,
            notes="BF16 fully resident + compile + cache_accel",
        )

    # ── BF16 + group_offload (streaming, same quality) ────────────────────────
    # Only need text_encoder + VAE + ~2 blocks + activations
    streaming_vram = te_fp8 + vae_size_mb + 2048 + activation_headroom  # FP8 encoder, streaming transformer
    if quality_tier and streaming_vram <= available_mb:
        return VRAMPlan(
            strategy=OffloadStrategy.GROUP_OFFLOAD,
            transformer_format=Format.BF16,
            text_encoder_format=Format.FP8,
            vae_format=Format.BF16,
            use_compile=False,
            use_cache_accel=False,
            use_group_offload=True,
            estimated_vram_mb=streaming_vram,
            notes="BF16 group_offload (streaming) + FP8 text encoder. No compile/cache (incompatible).",
        )

    # ── FP8 resident (half VRAM, hardware-accelerated) ────────────────────────
    total_fp8 = model_fp8 + te_fp8 + vae_size_mb + activation_headroom
    if total_fp8 <= available_mb:
        return VRAMPlan(
            strategy=OffloadStrategy.RESIDENT,
            transformer_format=Format.FP8,
            text_encoder_format=Format.FP8,
            vae_format=Format.BF16,
            use_compile=True,
            use_cache_accel=True,
            use_group_offload=False,
            estimated_vram_mb=total_fp8,
            notes="FP8 resident + compile + cache_accel",
        )

    # ── FP8 + group_offload (streaming FP8) ───────────────────────────────────
    fp8_streaming = te_fp8 + vae_size_mb + 1024 + activation_headroom
    if fp8_streaming <= available_mb:
        return VRAMPlan(
            strategy=OffloadStrategy.GROUP_OFFLOAD,
            transformer_format=Format.FP8,
            text_encoder_format=Format.FP8,
            vae_format=Format.BF16,
            use_compile=False,
            use_cache_accel=False,
            use_group_offload=True,
            estimated_vram_mb=fp8_streaming,
            notes="FP8 group_offload (streaming). No compile/cache.",
        )

    # ── model_cpu_offload + FP8 (component-level swap) ────────────────────────
    cpu_offload_vram = vae_size_mb + activation_headroom
    return VRAMPlan(
        strategy=OffloadStrategy.MODEL_CPU_OFFLOAD,
        transformer_format=Format.FP8,
        text_encoder_format=Format.FP8,
        vae_format=Format.BF16,
        use_compile=False,
        use_cache_accel=True,
        use_group_offload=False,
        estimated_vram_mb=cpu_offload_vram,
        notes="model_cpu_offload + FP8 + cache_accel. Components swap sequentially.",
    )


# ─── Apply Optimization Plan ───────────────────────────────────────────────────

def apply_vram_plan(pipe, plan: VRAMPlan, config) -> None:
    """Apply the VRAM optimization plan to a loaded pipeline.

    Args:
        pipe: The diffusers pipeline (already loaded from_pretrained)
        plan: The VRAMPlan to apply
        config: ModelConfig from the registry
    """
    # Step 1: Apply format (quantization) to components
    _apply_format(pipe, plan)

    # Step 2: Apply offload strategy
    _apply_offload(pipe, plan)

    # Step 3: Apply optimizations (compile, cache) if compatible
    _apply_optimizations(pipe, plan)


def _apply_format(pipe, plan: VRAMPlan) -> None:
    """Apply quantization format to pipeline components."""
    # Text encoder — quantize if plan says so
    if hasattr(pipe, "text_encoder") and plan.text_encoder_format in (Format.FP8, Format.INT8):
        _try_cast_to_fp8(pipe.text_encoder, "text_encoder")

    if hasattr(pipe, "text_encoder_2") and plan.text_encoder_format in (Format.FP8, Format.INT8):
        _try_cast_to_fp8(pipe.text_encoder_2, "text_encoder_2")

    # Transformer — quantize if plan says so (but respect precision_critical)
    if hasattr(pipe, "transformer") and plan.transformer_format == Format.FP8:
        _try_cast_to_fp8(pipe.transformer, "transformer")

    # VAE — ALWAYS keep in BF16, never quantize


def _try_cast_to_fp8(module, name: str) -> None:
    """Apply FP8 layerwise casting to a module."""
    try:
        module.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        logger.info("VRAM: %s cast to FP8", name)
    except AttributeError:
        logger.warning("VRAM: %s doesn't support layerwise_casting — staying BF16", name)
    except Exception as e:
        logger.warning("VRAM: %s FP8 cast failed (%s) — staying BF16", name, e)


def _apply_offload(pipe, plan: VRAMPlan) -> None:
    """Apply offload strategy to the pipeline."""
    if plan.strategy == OffloadStrategy.RESIDENT:
        pipe.to("cuda")
        logger.info("VRAM: pipeline resident on GPU")

    elif plan.strategy == OffloadStrategy.MODEL_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        logger.info("VRAM: model_cpu_offload enabled")

    elif plan.strategy == OffloadStrategy.GROUP_OFFLOAD:
        # Move text encoders + VAE to GPU (resident)
        for attr in ("text_encoder", "text_encoder_2", "vae"):
            if hasattr(pipe, attr):
                comp = getattr(pipe, attr)
                comp.to("cuda")
                if attr == "vae" and hasattr(comp, "enable_tiling"):
                    comp.enable_tiling()

        # Apply group_offload to transformer (block-level streaming)
        if hasattr(pipe, "transformer"):
            try:
                pipe.transformer.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=1,  # use_stream forces this in diffusers 0.37.0
                    use_stream=True,
                    record_stream=True,
                )
                logger.info("VRAM: transformer group_offload (use_stream=True)")
            except Exception as e:
                logger.warning("VRAM: group_offload failed (%s) — falling back to model_cpu_offload", e)
                pipe.enable_model_cpu_offload()


def _apply_optimizations(pipe, plan: VRAMPlan) -> None:
    """Apply compile and cache acceleration if compatible with the strategy."""
    # These are INCOMPATIBLE with group_offload — only apply on resident/cpu_offload paths
    if plan.use_group_offload:
        return

    # Cache acceleration (skip redundant denoising steps)
    if plan.use_cache_accel and hasattr(pipe, "transformer"):
        try:
            from diffusers import apply_first_block_cache, FirstBlockCacheConfig
            apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.05))
            logger.info("VRAM: first_block_cache enabled (threshold=0.05)")
        except ImportError:
            logger.debug("VRAM: first_block_cache not available")
        except Exception as e:
            logger.debug("VRAM: cache_accel failed (%s)", e)

    # torch.compile (regional — only repeated DiT blocks)
    if plan.use_compile and hasattr(pipe, "transformer"):
        try:
            pipe.transformer.compile_repeated_blocks(fullgraph=True)
            logger.info("VRAM: compile_repeated_blocks enabled")
        except Exception as e:
            logger.debug("VRAM: compile_repeated_blocks failed (%s)", e)


# ─── Cleanup ───────────────────────────────────────────────────────────────────

def release_vram():
    """Release all PyTorch CUDA memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
