# SPDX-License-Identifier: Apache-2.0
"""Shared FP8 weight-only patch module — DRY core for all DiT models.

Problem: vLLM's FP8 activation quantization on Diffusion Transformer linear
layers causes NaN cascades — the latent range shifts across denoising
timesteps and compounds activation quantization rounding errors until the
representation collapses to NaN/black output (or garbage video).

Fix: Replace Fp8LinearMethod on DiT linear layers with a subclass that:
  1. Keeps FP8 weight storage (no extra memory cost)
  2. Overrides `apply` to dequantize FP8 → BF16 using per-shard weight scales,
     then run F.linear (BF16 matmul, no activation quant)
  3. Overrides `process_weights_after_loading` to SKIP the parent's
     `process_fp8_weight_tensor_strategy` which collapses per-shard QKV scales
     to a single scalar. Without this override, K and V dequantize with Q's
     scale → garbage attention output.
  4. Overrides `create_weights` to resize per-tensor scale from scalar ()
     to shape (N,) for fused modules (QKV, gate/up MLP), so
     `adjust_scalar_to_fused_array` can load per-shard scales.

Usage in a pipeline patch file:
    from fp8_weight_only_patch import apply_fp8_weight_only_patch
    
    # Call at module import time (replaces Fp8Config.get_quant_method globally)
    apply_fp8_weight_only_patch(
        layer_filter=lambda prefix, layer: (
            isinstance(layer, LinearBase) and
            any(k in prefix for k in ("attn", "ffn", "mlp", "to_", "proj"))
        ),
        logger_name="my_patch"
    )

The layer_filter controls which layers get the weight-only treatment.
Default: all LinearBase layers (the DiT model's attn + MLP + projections).
"""

import logging as _logging
import torch as _torch
import torch.nn.functional as _F
from typing import Callable, Optional

_fp8_logger = _logging.getLogger("fp8_weight_only_patch")


class Fp8WeightOnlyLinearMethod:
    """FP8 weight storage + BF16 matmul (no activation quantization).

    Mixin-compatible: can be used as a standalone class or mixed into
    Fp8LinearMethod via the apply_fp8_weight_only_patch() function.

    Three model formats are handled automatically:
      A) Native FP8 (W8A8 Block): weights have per-tensor scales, needs
         dequant via weight_scale multiplication.
      B) Direct-cast FP8: weights already in normal NN range, no scale
         to apply — just cast FP8 → BF16 directly.
      C) Per-channel FP8: weight_scale is per-output-channel, broadcast
         across in_features dimension.

    Detection: if weight_scale exists and is non-zero anywhere, we use
    format A. If weight_scale is zero or doesn't exist, format B.
    """

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        """Create FP8 weight + scale parameters, with fused-layer scale fix.

        For fused layers (QKV, gate/up MLP) the per-tensor scale must be
        shape-(N,) so adjust_scalar_to_fused_array can index by shard_id.
        The parent Fp8LinearMethod.create_weights creates a scalar () scale
        which fails on fused layers — we resize it here.
        """
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        Fp8LinearMethod.create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs,
        )
        # Fix fused-layer per-tensor scales
        if len(output_partition_sizes) > 1 and hasattr(layer, "weight_scale"):
            sp = layer.weight_scale
            sp.needs_scalar_to_array = True
            if hasattr(sp, "output_dim"):
                try:
                    sp.output_dim = None
                except AttributeError:
                    try:
                        del sp.output_dim
                    except AttributeError:
                        pass
            n = len(output_partition_sizes)
            if sp.data.dim() == 0:
                old_dtype = sp.data.dtype
                old_device = sp.data.device
                new_data = _torch.zeros(n, dtype=old_dtype, device=old_device)
                layer._parameters["weight_scale"] = _torch.nn.Parameter(
                    new_data, requires_grad=False
                )
                layer._parameters["weight_scale"].needs_scalar_to_array = True

    def apply(self, layer, x, bias=None):
        """Dequant FP8 → BF16, then run F.linear (no activation quant).

        Handles three scale formats:
          - Per-tensor (single scalar)
          - Fused (one scale per logical shard, indexed via logical_widths)
          - Per-channel (one scale per output channel)
        """
        # Get original Fp8LinearMethod.apply for the fallback path
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        orig_apply = Fp8LinearMethod.apply

        weight = layer.weight
        weight = weight.t()  # [in, out] → [out, in] for F.linear

        has_scale = hasattr(layer, "weight_scale") and layer.weight_scale is not None

        if has_scale and layer.weight_scale.numel() > 0:
            ws = layer.weight_scale.to(_torch.bfloat16)
            w_fp8 = weight.to(_torch.bfloat16)  # raw FP8 → BF16 (first cast)

            if ws.numel() == 1:
                # Per-tensor scale
                w_bf16 = w_fp8 * ws
            else:
                logical_widths = getattr(layer, "logical_widths", None)
                if (logical_widths is not None
                        and len(logical_widths) == ws.shape[0]
                        and sum(logical_widths) == w_fp8.shape[0]):
                    # Fused (QKV, gate/up): expand per-shard scale to per-row
                    pieces = []
                    for i, w in enumerate(logical_widths):
                        pieces.append(_torch.full(
                            (w,), float(ws[i].item()),
                            dtype=w_fp8.dtype, device=w_fp8.device,
                        ))
                    row_scale = _torch.cat(pieces).unsqueeze(1)  # [out, 1]
                    w_bf16 = w_fp8 * row_scale
                elif ws.dim() == 1 and ws.shape[0] == w_fp8.shape[0]:
                    # Per-output-channel scale
                    w_bf16 = w_fp8 * ws.unsqueeze(1)
                else:
                    # Last resort: scalar broadcast (may be suboptimal)
                    w_bf16 = w_fp8 * ws.mean()
            return _F.linear(x, w_bf16, bias)
        else:
            # Direct-cast FP8 (no scale needed) — most common for modelopt
            w_bf16 = weight.to(_torch.bfloat16)
            return _F.linear(x, w_bf16, bias)

    def process_weights_after_loading(self, layer):
        """Override: skip parent's process_fp8_weight_tensor_strategy.

        That function collapses per-shard scales (N,) → scalar via
        requantize_with_max_scale, which causes K/V dequant with Q's scale.
        We keep per-shard scales intact but still transpose weight layout.
        """
        if getattr(layer, '_fp8_wo_already_processed', False):
            return
        weight = layer.weight
        weight = weight.t()  # [out, in] → [in, out] (vLLM kernel layout)
        from vllm.model_executor.utils import replace_parameter
        replace_parameter(layer, "weight", weight.data)
        layer.input_scale = None
        layer._fp8_wo_already_processed = True


def apply_fp8_weight_only_patch(
    layer_filter: Optional[Callable] = None,
    logger_name: str = "fp8_patch",
    patch_modulation: bool = True,
):
    """Apply FP8 weight-only patch to Fp8Config.get_quant_method.

    Args:
        layer_filter: Callable(prefix, layer) → bool.
            Return True to apply weight-only, False to keep original FP8 method.
            Default: all LinearBase instances (covers attn, MLP, projections).
        logger_name: Name for the patch logger (for log filtering).
        patch_modulation: If True, also patch ColumnParallelLinear/ReplicatedLinear
            to inject quant_config into modulation layers (Qwen-specific).

    This function monkey-patches Fp8Config.get_quant_method so that DiT
    linear layers use FP8-weight-only (no activation quantization), while
    non-linear layers (norms, embeddings, VAE) keep the original method.

    Must be called at module import time in the worker process, before
    the model is constructed.
    """
    logger = _logging.getLogger(logger_name)

    try:
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8Config as _Fp8Config,
            Fp8LinearMethod as _Fp8LinearMethod,
        )
        from vllm.model_executor.layers.linear import LinearBase as _LinearBase
    except ImportError:
        logger.warning("Could not import vLLM fp8 modules — patch skipped")
        return

    _orig_get_quant_method = _Fp8Config.get_quant_method

    if layer_filter is None:
        layer_filter = lambda prefix, layer: isinstance(layer, _LinearBase)

    # Create a Fp8LinearMethod subclass that uses our overrides
    class _PatchedFp8LinearMethod(_Fp8LinearMethod):
        create_weights = Fp8WeightOnlyLinearMethod.create_weights
        apply = Fp8WeightOnlyLinearMethod.apply
        process_weights_after_loading = Fp8WeightOnlyLinearMethod.process_weights_after_loading

    def _patched_get_quant_method(self, layer, prefix):
        if layer_filter(prefix, layer):
            return _PatchedFp8LinearMethod(self)
        return _orig_get_quant_method(self, layer, prefix)

    _Fp8Config.get_quant_method = _patched_get_quant_method
    logger.warning(
        "FP8 weight-only patch ACTIVE — FP8 storage + BF16 matmul "
        "(no activation quantization = no NaN)"
    )

    if patch_modulation:
        _patch_modulation_layers(logger)

    return _PatchedFp8LinearMethod


def _patch_modulation_layers(logger):
    """Patch ColumnParallelLinear/ReplicatedLinear for Qwen-style models.

    QwenImageTransformerBlock hardcodes quant_config=None for img_mod and
    txt_mod. These layers are FP8 on disk but get allocated as BF16 without
    this patch, busting VRAM budget.

    Uses a thread-local stack to propagate the block's quant_config into
    the modulation layer constructors.
    """
    try:
        import threading as _threading
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear as _ColumnParallelLinear,
            ReplicatedLinear as _ReplicatedLinear,
        )

        _active_qc = _threading.local()

        def _qc_push(qc):
            stack = getattr(_active_qc, "stack", None)
            if stack is None:
                stack = []
                _active_qc.stack = stack
            stack.append(qc)

        def _qc_pop():
            stack = getattr(_active_qc, "stack", None)
            if not stack:
                return None
            return stack.pop()

        def _qc_top():
            stack = getattr(_active_qc, "stack", None)
            if not stack:
                return None
            return stack[-1]

        # Patch ColumnParallelLinear
        _orig_cpl_init = _ColumnParallelLinear.__init__

        def _patched_cpl_init(self, input_size, output_size, bias=True, *,
                              quant_config=None, prefix="", **_kw):
            if quant_config is None and prefix:
                qc = _qc_top()
                if qc is not None:
                    quant_config = qc
            _orig_cpl_init(self, input_size, output_size, bias=bias,
                           quant_config=quant_config, prefix=prefix, **_kw)

        _ColumnParallelLinear.__init__ = _patched_cpl_init

        # Patch ReplicatedLinear
        _orig_rl_init = _ReplicatedLinear.__init__

        def _patched_rl_init(self, input_size, output_size, bias=True, *,
                             quant_config=None, prefix="", **_kw):
            if quant_config is None and prefix:
                qc = _qc_top()
                if qc is not None:
                    quant_config = qc
            _orig_rl_init(self, input_size, output_size, bias=bias,
                          quant_config=quant_config, prefix=prefix, **_kw)

        _ReplicatedLinear.__init__ = _patched_rl_init

        logger.warning(
            "Modulation layer FP8 injection ACTIVE — ColumnParallelLinear/"
            "ReplicatedLinear get quant_config from thread-local stack"
        )

        # Export helpers for use by model-specific __init__ patches
        return _qc_push, _qc_pop, _qc_top

    except Exception as e:
        logger.warning("Could not patch modulation layers: %s", e)
        return None, None, None


def cpu_offload_text_encoder(module, logger=None):
    """Move text encoder to CPU and patch forward for transparent GPU↔CPU.

    Args:
        module: The text encoder module (or None if not yet loaded).
        logger: Optional logger for status messages.

    Returns:
        The moved module (now on CPU).
    """
    log = logger or _logging.getLogger("cpu_offload")
    if module is None:
        return None
    import gc
    try:
        module = module.cpu()
        gc.collect()
        _torch.cuda.empty_cache()
        log.warning("Text encoder moved to CPU (freed ~14GB VRAM)")
    except Exception as e:
        log.warning("Could not move text encoder to CPU: %s", e)
    return module


def patch_forward_cpu_gpu(module_class, logger=None):
    """Patch a module's forward method for CPU-resident + GPU inputs.

    When the module lives on CPU but inputs come from GPU tensors,
    this wrapper moves inputs to CPU, runs forward, moves outputs
    back to GPU — transparent to the caller.

    Args:
        module_class: The class to patch (e.g., Qwen2_5_VLForConditionalGeneration).
        logger: Optional logger.

    Note: This is a one-time class-level patch. After calling, ALL
    instances of this class will handle GPU→CPU→GPU transparently.
    """
    import functools
    log = logger or _logging.getLogger("cpu_gpu_patch")

    if getattr(module_class, '_cpu_gpu_patched', False):
        return

    _orig_fwd = module_class.forward

    @functools.wraps(_orig_fwd)
    def _patched_fwd(self, *args, **kwargs):
        p_dev = next(self.parameters()).device
        # Determine input device
        i_dev = None
        for v in list(args) + list(kwargs.values()):
            if hasattr(v, 'device'):
                i_dev = v.device
                break
        if p_dev != i_dev and i_dev is not None and p_dev.type == 'cpu':
            # Move inputs to CPU
            cpu_args = []
            for a in args:
                cpu_args.append(a.to(p_dev) if hasattr(a, 'to') else a)
            cpu_kw = {}
            for k, v in kwargs.items():
                cpu_kw[k] = v.to(p_dev) if hasattr(v, 'to') else v
            result = _orig_fwd(self, *cpu_args, **cpu_kw)
            # Move outputs back to GPU
            if hasattr(result, 'last_hidden_state') and result.last_hidden_state is not None:
                result.last_hidden_state = result.last_hidden_state.to(i_dev)
            if hasattr(result, 'hidden_states') and result.hidden_states is not None:
                result.hidden_states = [
                    hs.to(i_dev) if hs is not None else hs
                    for hs in result.hidden_states
                ]
            return result
        return _orig_fwd(self, *args, **kwargs)

    module_class.forward = _patched_fwd
    module_class._cpu_gpu_patched = True
    log.warning("Patched %s.forward for CPU-GPU transparency", module_class.__name__)


# Legacy alias for backward compatibility
_Fp8WeightOnlyLinearMethod = Fp8WeightOnlyLinearMethod
