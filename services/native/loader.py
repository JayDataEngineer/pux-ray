"""Adaptive VRAM loader — replaces mmGP with native diffusers APIs.

This is our custom VRAM management package. It uses:
  - enable_group_offload(use_stream=True) for block-level streaming
  - enable_layerwise_casting(fp8) for on-the-fly quantization
  - compile_repeated_blocks for kernel fusion
  - apply_first_block_cache for step skipping
  - PEFT for LoRA management

The loader inspects available VRAM and selects the optimal strategy
automatically — no user configuration needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class Strategy(str, Enum):
    """VRAM management strategy."""
    RESIDENT = "resident"            # Everything in VRAM — fastest
    GROUP_OFFLOAD = "group_offload"  # Stream blocks — like mmGP
    CPU_OFFLOAD = "cpu_offload"      # Component swap — fallback
    FP8_RESIDENT = "fp8_resident"    # FP8 quantized, resident
    FP8_OFFLOAD = "fp8_offload"      # FP8 quantized, streaming


@dataclass
class VRAMPlan:
    """The optimization plan selected by the adaptive loader."""
    strategy: Strategy
    use_compile: bool
    use_cache: bool
    use_fp8: bool
    vram_estimate_mb: int
    notes: str = ""


def _available_vram_mb() -> int:
    """Get usable VRAM (85% of free, leaving activation headroom)."""
    try:
        free, _ = torch.cuda.mem_get_info(0)
        return int(free / (1024 * 1024) * 0.85)
    except Exception:
        return 0


def _estimate_params_mb(module) -> int:
    """Estimate module size in MB (BF16)."""
    if module is None:
        return 0
    params = sum(p.numel() for p in module.parameters())
    return int(params * 2 / (1024 * 1024))


def plan(pipe) -> VRAMPlan:
    """Analyze a loaded pipeline and select the optimal VRAM strategy.

    Called after from_pretrained() but before generation.
    Inspects model sizes and available VRAM to choose:
    - Resident (fits entirely)
    - Group offload (streaming — like mmGP)
    - FP8 (halve the weights)
    - CPU offload (last resort)
    """
    available = _available_vram_mb()

    # Measure actual component sizes
    transformer_mb = 0
    encoder_mb = 0
    vae_mb = 0

    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        transformer_mb = _estimate_params_mb(pipe.transformer)
    elif hasattr(pipe, "unet") and pipe.unet is not None:
        transformer_mb = _estimate_params_mb(pipe.unet)

    for attr in ("text_encoder", "text_encoder_2", "tokenizer"):
        if hasattr(pipe, attr):
            comp = getattr(pipe, attr)
            if comp is not None and hasattr(comp, "parameters"):
                encoder_mb += _estimate_params_mb(comp)

    if hasattr(pipe, "vae") and pipe.vae is not None:
        vae_mb = _estimate_params_mb(pipe.vae)

    activation_mb = 4096  # Reserve 4GB for activations
    total_bf16 = transformer_mb + encoder_mb + vae_mb + activation_mb
    total_fp8 = transformer_mb // 2 + encoder_mb // 2 + vae_mb + activation_mb
    streaming_mb = encoder_mb + vae_mb + 2048 + activation_mb  # Only 2 blocks resident

    logger.info("VRAM plan: available=%dMB, transformer=%dMB, encoder=%dMB, vae=%dMB",
                available, transformer_mb, encoder_mb, vae_mb)

    # ── Decision tree ──────────────────────────────────────────────────────────

    # 1. BF16 resident — best case
    if total_bf16 <= available:
        return VRAMPlan(
            strategy=Strategy.RESIDENT,
            use_compile=True,
            use_cache=True,
            use_fp8=False,
            vram_estimate_mb=total_bf16,
            notes=f"BF16 resident ({total_bf16}MB ≤ {available}MB available) + compile + cache",
        )

    # 2. BF16 group_offload — stream blocks (like mmGP)
    if streaming_mb <= available:
        return VRAMPlan(
            strategy=Strategy.GROUP_OFFLOAD,
            use_compile=False,
            use_cache=False,
            use_fp8=False,
            vram_estimate_mb=streaming_mb,
            notes=f"BF16 group_offload streaming ({streaming_mb}MB) — no compile/cache",
        )

    # 3. FP8 resident — halve weights
    if total_fp8 <= available:
        return VRAMPlan(
            strategy=Strategy.FP8_RESIDENT,
            use_compile=True,
            use_cache=True,
            use_fp8=True,
            vram_estimate_mb=total_fp8,
            notes=f"FP8 resident ({total_fp8}MB) + compile + cache",
        )

    # 4. FP8 group_offload — stream FP8 blocks
    fp8_stream = encoder_mb // 2 + vae_mb + 1024 + activation_mb
    if fp8_stream <= available:
        return VRAMPlan(
            strategy=Strategy.FP8_OFFLOAD,
            use_compile=False,
            use_cache=False,
            use_fp8=True,
            vram_estimate_mb=fp8_stream,
            notes=f"FP8 group_offload streaming ({fp8_stream}MB)",
        )

    # 5. CPU offload — last resort
    return VRAMPlan(
        strategy=Strategy.CPU_OFFLOAD,
        use_compile=False,
        use_cache=True,
        use_fp8=False,
        vram_estimate_mb=vae_mb + activation_mb,
        notes="model_cpu_offload fallback",
    )


def apply(pipe, vram_plan: VRAMPlan) -> None:
    """Apply the VRAM plan to a loaded pipeline.

    This replaces mmGP's offload.all() with native diffusers APIs.
    """
    s = vram_plan.strategy

    if s == Strategy.RESIDENT:
        pipe.to("cuda")
        _apply_optimizations(pipe, vram_plan)

    elif s == Strategy.GROUP_OFFLOAD:
        _apply_streaming(pipe, use_fp8=False)
        # No compile/cache on streaming path (incompatible, verified)

    elif s == Strategy.FP8_RESIDENT:
        _apply_fp8(pipe)
        pipe.to("cuda")
        _apply_optimizations(pipe, vram_plan)

    elif s == Strategy.FP8_OFFLOAD:
        _apply_fp8(pipe)
        _apply_streaming(pipe, use_fp8=True)

    elif s == Strategy.CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        _apply_optimizations(pipe, vram_plan)

    logger.info("VRAM applied: %s", vram_plan.notes)


def _apply_fp8(pipe) -> None:
    """Apply FP8 layerwise casting to transformer and text encoders."""
    for attr in ("text_encoder", "text_encoder_2"):
        if hasattr(pipe, attr):
            comp = getattr(pipe, attr)
            if comp is not None:
                try:
                    comp.enable_layerwise_casting(
                        storage_dtype=torch.float8_e4m3fn,
                        compute_dtype=torch.bfloat16,
                    )
                    logger.info("FP8: %s cast", attr)
                except Exception as e:
                    logger.debug("FP8: %s skip (%s)", attr, e)

    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        try:
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            logger.info("FP8: transformer cast")
        except Exception as e:
            logger.debug("FP8: transformer skip (%s)", e)


def _apply_streaming(pipe, use_fp8: bool = False) -> None:
    """Apply group_offload streaming — text encoders + VAE resident,
    transformer blocks stream via CUDA streams."""
    # Text encoders stay on GPU
    for attr in ("text_encoder", "text_encoder_2"):
        if hasattr(pipe, attr):
            comp = getattr(pipe, attr)
            if comp is not None:
                comp.to("cuda")

    # VAE on GPU with tiling
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to("cuda")
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    # Transformer streams via group_offload
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if transformer is not None:
        try:
            transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,  # use_stream forces this in diffusers 0.37.0
                use_stream=True,
                record_stream=True,
            )
            logger.info("Streaming: transformer group_offload (use_stream=True)")
        except Exception as e:
            logger.warning("Streaming: group_offload failed (%s) — falling back", e)
            pipe.enable_model_cpu_offload()


def _apply_optimizations(pipe, vram_plan: VRAMPlan) -> None:
    """Apply compile + cache acceleration (only on resident/cpu_offload paths)."""
    # Cache acceleration — skip redundant denoising steps
    if vram_plan.use_cache and hasattr(pipe, "transformer") and pipe.transformer:
        try:
            from diffusers import apply_first_block_cache, FirstBlockCacheConfig
            apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.05))
            logger.info("Optimization: first_block_cache enabled")
        except Exception:
            pass

    # Regional compilation — fuse DiT block kernels
    if vram_plan.use_compile and hasattr(pipe, "transformer") and pipe.transformer:
        try:
            pipe.transformer.compile_repeated_blocks(fullgraph=True)
            logger.info("Optimization: compile_repeated_blocks enabled")
        except Exception:
            pass


def release():
    """Release all CUDA memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
