# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ════════════════════════════════════════════════════════════════════════
# FP8 WEIGHT-ONLY PATCH — using shared DRY module (scripts/fp8_weight_only_patch.py)
# ════════════════════════════════════════════════════════════════════════
# Import the shared FP8 weight-only patch and apply with model-specific
# layer filter. This replaces ALL DiT linear layers with dequant→BF16.
#
# The shared module handles:
#   - Fp8Config.get_quant_method monkey-patch
#   - _Fp8WeightOnlyLinearMethod (create_weights, apply, process_weights_after_loading)
#   - Fused-layer scale resizing for QKV/GateUp projections
#   - CPU-GPU forward transparency
#
# Model-specific code below:
#   - Modulation layer quant_config injection (QwenImageTransformerBlock)
#   - Text encoder CPU offload (Qwen2_5_VLForConditionalGeneration)
#   - Full pipeline class (preserved from original)
# ════════════════════════════════════════════════════════════════════════
import logging as _logging
_fp8_patch_logger = _logging.getLogger("fp8_qwen_image_edit_patch")

try:
    from fp8_weight_only_patch import (
        apply_fp8_weight_only_patch,
    )
    from vllm.model_executor.layers.linear import LinearBase as _LinearBase

    # Apply shared FP8 weight-only patch to ALL DiT linear layers
    apply_fp8_weight_only_patch(
        layer_filter=lambda prefix, layer: isinstance(layer, _LinearBase),
        logger_name="fp8_qwen_image_edit_patch",
        patch_modulation=False,  # We handle modulation separately below
    )

    _fp8_patch_logger.warning(
        "FP8 weight-only patch applied (via shared module) on "
        "Qwen-Image-Edit DiT — FP8 storage + BF16 matmul"
    )
except Exception as _e:
    _fp8_patch_logger.warning("Could not apply FP8 weight-only patch: %s", _e)
    import traceback as _tb
    _fp8_patch_logger.warning(_tb.format_exc())

# ════════════════════════════════════════════════════════════════════════
# FP8 FOR MODULATION LAYERS (img_mod, txt_mod)
# ════════════════════════════════════════════════════════════════════════
# QwenImageTransformerBlock hardcodes `quant_config=None` for img_mod and
# txt_mod (ColumnParallelLinear).  But on disk these layers ARE FP8 (the
# ComfyUI checkpoint quantized everything).  Without this patch, img_mod
# and txt_mod each allocate 113 MB BF16 per block × 60 blocks = 13.5 GB,
# busting the 24 GB budget.
#
# We can't simply pass quant_config because the hardcoded `=None` overrides
# it.  Instead, monkey-patch ColumnParallelLinear.__init__ to inject the
# block's quant_config when: (a) the caller passed None, and (b) the prefix
# matches a modulation layer inside a transformer_block.
#
# A thread-local carries the active block's quant_config so the deep
# ColumnParallelLinear constructor can pick it up.
try:
    import threading as _threading
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear as _ColumnParallelLinear,
        LinearBase as _LinearBase2,
        ReplicatedLinear as _ReplicatedLinear,
    )
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel as _Transformer2D,
        QwenImageTransformerBlock as _TransformerBlock,
    )

    _active_block_qc = _threading.local()

    def _qc_push(qc):
        """Push a quant_config onto the thread-local stack."""
        stack = getattr(_active_block_qc, "stack", None)
        if stack is None:
            stack = []
            _active_block_qc.stack = stack
        stack.append(qc)

    def _qc_pop():
        """Pop the top quant_config from the stack (returns None if empty)."""
        stack = getattr(_active_block_qc, "stack", None)
        if not stack:
            return None
        return stack.pop()

    def _qc_top():
        """Read the top of the quant_config stack (returns None if empty)."""
        stack = getattr(_active_block_qc, "stack", None)
        if not stack:
            return None
        return stack[-1]

    _orig_col_pll_init = _ColumnParallelLinear.__init__

    def _patched_col_pll_init(
        self,
        input_size,
        output_size,
        bias=True,
        *,
        quant_config=None,
        prefix="",
        **_kw,
    ):
        if (
            quant_config is None
            and prefix
            and ("img_mod" in prefix or "txt_mod" in prefix)
        ):
            qc = _qc_top()
            if qc is not None:
                quant_config = qc
                _fp8_patch_logger.debug(
                    "Injected quant_config into %s", prefix
                )
        _orig_col_pll_init(
            self,
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
            **_kw,
        )

    _ColumnParallelLinear.__init__ = _patched_col_pll_init

    # ReplicatedLinear is used for img_in/txt_in/norm_out.linear/proj_out
    # which are also hardcoded `quant_config=None` but stored FP8 on disk.
    # Also covers timestep_embedder.linear_{1,2} inside QwenTimestepProjEmbeddings.
    _orig_repl_init = _ReplicatedLinear.__init__

    def _patched_repl_init(
        self,
        input_size,
        output_size,
        bias=True,
        *,
        quant_config=None,
        prefix="",
        **_kw,
    ):
        if (
            quant_config is None
            and prefix
            and any(
                p in prefix
                for p in (
                    "img_in", "txt_in", "norm_out", "proj_out",
                    "timestep_embedder",
                )
            )
        ):
            qc = _qc_top()
            if qc is not None:
                quant_config = qc
                _fp8_patch_logger.debug(
                    "Injected quant_config into ReplicatedLinear %s", prefix
                )
        _orig_repl_init(
            self,
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
            **_kw,
        )

    _ReplicatedLinear.__init__ = _patched_repl_init

    _orig_block_init = _TransformerBlock.__init__

    def _patched_block_init(self, *args, quant_config=None, **kwargs):
        _qc_push(quant_config)
        try:
            _orig_block_init(self, *args, quant_config=quant_config, **kwargs)
        finally:
            _qc_pop()

    _TransformerBlock.__init__ = _patched_block_init

    # Also set thread-local during the outer transformer model __init__
    # so ReplicatedLinear children (img_in/txt_in/norm_out.linear/proj_out)
    # see the quant_config.  Uses a STACK so that nested QwenImageTransformerBlock
    # __init__ calls don't clobber the outer QwenImageTransformer2DModel setting
    # (which would otherwise be cleared before norm_out.linear is constructed).
    _orig_tf2d_init = _Transformer2D.__init__

    def _patched_tf2d_init(self, *args, od_config=None, **kwargs):
        qc = getattr(od_config, "quantization_config", None)
        _qc_push(qc)
        try:
            _orig_tf2d_init(self, *args, od_config=od_config, **kwargs)
        finally:
            _qc_pop()

    _Transformer2D.__init__ = _patched_tf2d_init

    _fp8_patch_logger.warning(
        "Patched ColumnParallelLinear + ReplicatedLinear + "
        "QwenImageTransformer{Block,2DModel} to inject FP8 quant_config "
        "into img_mod/txt_mod/img_in/txt_in/norm_out/proj_out"
    )
except Exception as _e:
    _fp8_patch_logger.warning("Could not apply mod-layer FP8 patch: %s", _e)
    import traceback as _tb
    _fp8_patch_logger.warning(_tb.format_exc())

# ════════════════════════════════════════════════════════════════════════
# TEXT ENCODER CPU OFFLOAD
# ════════════════════════════════════════════════════════════════════════
# Move the text encoder to CPU to free ~14GB VRAM for the DiT.
# Patch its forward method so CPU-resident parameters receive GPU inputs
# gracefully (move inputs to CPU, run, move outputs back to GPU).
# ════════════════════════════════════════════════════════════════════════
import functools as _ft


def _patch_qwen_vl_forward():
    """Patch Qwen2_5_VLForConditionalGeneration.forward for CPU text encoder.

    When the text encoder lives on CPU but inputs come from GPU, this
    moves inputs to CPU, runs the forward pass, and moves outputs back
    to GPU — transparent to the caller.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVL

    if getattr(_QwenVL, '_patched_fwd', False):
        return

    _orig_fwd = _QwenVL.forward

    @_ft.wraps(_orig_fwd)
    def _patched_fwd(self, input_ids=None, attention_mask=None,
                     pixel_values=None, image_grid_thw=None,
                     **kwargs):
        p_dev = next(self.parameters()).device  # CPU
        # Determine input device from first non-None input
        for inp in (input_ids, attention_mask, pixel_values):
            if inp is not None:
                i_dev = inp.device
                break
        else:
            return _orig_fwd(self, input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             image_grid_thw=image_grid_thw,
                             **kwargs)

        if p_dev != i_dev and i_dev is not None:
            with _torch.device(p_dev):
                result = _orig_fwd(
                    self,
                    input_ids=input_ids.to(p_dev) if input_ids is not None else None,
                    attention_mask=attention_mask.to(p_dev) if attention_mask is not None else None,
                    pixel_values=pixel_values.to(p_dev) if pixel_values is not None else None,
                    image_grid_thw=image_grid_thw.to(p_dev) if image_grid_thw is not None else None,
                    **kwargs,
                )
                # Move hidden states back to GPU
                if hasattr(result, 'hidden_states') and result.hidden_states is not None:
                    result.hidden_states = [
                        hs.to(i_dev) if hs is not None else hs
                        for hs in result.hidden_states
                    ]
                if hasattr(result, 'last_hidden_state') and result.last_hidden_state is not None:
                    result.last_hidden_state = result.last_hidden_state.to(i_dev)
                return result
        return _orig_fwd(self, input_ids=input_ids,
                         attention_mask=attention_mask,
                         pixel_values=pixel_values,
                         image_grid_thw=image_grid_thw,
                         **kwargs)

    _QwenVL.forward = _patched_fwd
    _QwenVL._patched_fwd = True


_patch_qwen_vl_forward()

# ════════════════════════════════════════════════════════════════════════
# ORIGINAL PIPELINE (preserved with text encoder CPU offload)
# ════════════════════════════════════════════════════════════════════════
# The rest of this file is the original pipeline_qwen_image_edit_plus.py
# with TWO changes:
#   1. Text encoder is moved to .cpu() after loading (saves ~14GB VRAM)
#   2. Layerwise offload is enabled for the DiT
# Everything else is identical to the original.
# ════════════════════════════════════════════════════════════════════════

import json
import logging
import os
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import calculate_shift
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    calculate_dimensions,
    retrieve_latents,
    retrieve_timesteps,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES = QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES


def get_qwen_image_edit_plus_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageEditPlusPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)
    latent_channels = vae_config.get("z_dim", 16)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImageEditPlusPipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                continue

            if not isinstance(raw_image, list):
                raw_image = [raw_image]
            if len(raw_image) > MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES:
                raise ValueError(
                    f"Received {len(raw_image)} input images. "
                    f"At most {MAX_QWEN_IMAGE_EDIT_PLUS_INPUT_IMAGES} images are supported by this model."
                )
            image = [
                PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image | np.ndarray | torch.Tensor, im)
                for im in raw_image
            ]

            # Calculate dimensions based on first image
            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = request.sampling_params.height or calculated_height
            width = request.sampling_params.width or calculated_width

            height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)

            prompt["additional_information"]["calculated_height"] = calculated_height
            prompt["additional_information"]["calculated_width"] = calculated_width
            request.sampling_params.height = height
            request.sampling_params.width = width

            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []

            for img in image:
                if isinstance(img, torch.Tensor) and len(img.shape) > 1 and img.shape[1] == latent_channels:
                    continue

                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))

                condition_images.append(image_processor.resize(img, condition_height, condition_width))
                vae_images.append(image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

            prompt["additional_information"]["condition_images"] = condition_images
            prompt["additional_information"]["vae_images"] = vae_images
            prompt["additional_information"]["condition_image_sizes"] = condition_image_sizes
            prompt["additional_information"]["vae_image_sizes"] = vae_image_sizes
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_qwen_image_edit_plus_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImageEditPlusPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


class QwenImageEditPlusPipeline(
    nn.Module, SupportImageInput, QwenImageCFGParallelMixin, DiffusionPipelineProfilerMixin
):
    """Qwen-Image-Edit-2511 pipeline with FP8 weight-only + CPU text encoder.

    Changes from original:
      - Text encoder loaded on CPU (saves ~14GB VRAM)
      - Uses FP8 weight-only for DiT attn+mlp layers (monkey-patched above)
      - Removed --enable-layerwise-offload dependency
      - Uses --quantization fp8 + patch for NaN-free inference

    VRAM budget on 24GB RTX 4090:
      DiT (20B FP8 weight-only):  ~20 GB  (all 60 blocks on GPU)
      VAE (with tiling):           ~0.3 GB
      Activations:                  ~3 GB
      Text encoder (CPU):          ~0 GB  (moved to RAM after prefill)
      ────────────────────────────────────
      Total:                      ~23 GB  ✓ fits on 24GB
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.device = get_local_device()
        model = od_config.model

        local_files_only = os.path.isdir(model)

        qwen_subfolders = ["scheduler", "text_encoder", "vae", "tokenizer", "processor"]
        prefetch_subfolders(
            model,
            qwen_subfolders,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # ── TEXT ENCODER ON CPU (saves ~16GB VRAM for the DiT) ──────────
        # CRITICAL: Load directly on CPU — NOT via .cpu() after GPU load.
        # The parent caller wraps __init__ in `with target_device:` (CUDA),
        # so by default from_pretrained allocates on GPU first.  For a 16GB
        # text encoder that competes with the 20GB DiT for a 24GB card,
        # the transient GPU copy causes OOM during DiT construction even
        # though .cpu() frees it to the caching allocator afterwards
        # (the allocator retains the memory and DiT blocks need it back as
        # FP8 weights, but by then fragmentation + overlap kills us).
        # Loading directly on CPU avoids the GPU roundtrip entirely.
        import gc as _gc
        with _torch.device("cpu"):
            self.text_encoder = from_pretrained_with_prefetch(
                Qwen2_5_VLForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            )
        _gc.collect()
        _torch.cuda.empty_cache()

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLQwenImage.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        ).to(self.device)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        # ── BUGFIX: Inject quant_config from od_config ──────────────────────
        # get_transformer_config_kwargs calls tf_model_config.to_dict() which
        # returns *only* self.params (the raw JSON keys).  The raw JSON has
        # "quantization_config" (a dict), but QwenImageTransformer2DModel
        # expects "quant_config" (a QuantizationConfig object).  Without this
        # fix, quant_config defaults to None → every linear layer allocates
        # BF16 weights → GPU OOM (20B model ~40GB on a 24GB card).
        #
        # od_config.quantization_config was already resolved to a
        # QuantizationConfig instance by _propagate_quantization_from_tf_config
        # inside enrich_config() — we just need to pass it along.
        if "quant_config" not in transformer_kwargs:
            qc = getattr(od_config, "quantization_config", None)
            if qc is not None and not isinstance(qc, (str, dict)):
                transformer_kwargs["quant_config"] = qc
        # ─────────────────────────────────────────────────────────────────────
        self.transformer = QwenImageTransformer2DModel(od_config=od_config, **transformer_kwargs)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.processor = from_pretrained_with_prefetch(
            Qwen2VLProcessor.from_pretrained,
            model,
            subfolder="processor",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain how the user's "
            "text instruction should alter or modify the image. Generate a new image that meets "
            "the user's requirements while maintaining consistency with the original input where "
            "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ── All methods below are identical to the original pipeline ─────────

    def check_inputs(
        self,
        prompt,
        height,
        width,
        image=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `prompt_embeds_mask` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > self.tokenizer_max_length:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than {self.tokenizer_max_length} but is {max_sequence_length}"
            )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        image: list[torch.Tensor] | torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        template_tokens = self.tokenizer(
            [template.format(base_img_prompt)],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[torch.Tensor] | torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                image,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std
        return image_latents

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")
        if negative_prompt is None:
            logger.warning(
                "negative_prompt is not set. The official Qwen-Image-Edit model "
                "may produce lower-quality results without a negative_prompt. "
                "Qwen official repository recommends to use whitespace string as negative_prompt. "
                "Note: some distilled variants may not be affected by this."
            )

        if (
            not isinstance(first_prompt, str)
            and "vae_images" in (additional_information := first_prompt.get("additional_information", {}))
            and "condition_images" in additional_information
        ):
            condition_images = additional_information.get("condition_images")
            vae_images = additional_information.get("vae_images")
            condition_image_sizes = additional_information.get("condition_image_sizes")
            vae_image_sizes = additional_information.get("vae_image_sizes")
            calculated_height = additional_information.get("calculated_height")
            calculated_width = additional_information.get("calculated_width")
            height = req.sampling_params.height
            width = req.sampling_params.width
        else:
            if image is None:
                raise ValueError("Image is required for QwenImageEditPlusPipeline")

            if not isinstance(image, list):
                image = [image]

            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = height or calculated_height
            width = width or calculated_width

            height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)

            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []

            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        true_cfg_scale = req.sampling_params.true_cfg_scale or true_cfg_scale
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        self.check_inputs(
            prompt,
            height,
            width,
            image,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            image=condition_images,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                image=condition_images,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        latents = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            image_latents=image_latents,
            cfg_normalize=True,
            additional_transformer_kwargs={
                "return_dict": False,
                "attention_kwargs": self.attention_kwargs,
            },
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

