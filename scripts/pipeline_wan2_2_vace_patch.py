# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════
# FP8 WEIGHT-ONLY PATCH (FP8 storage, BF16 matmul)
# ════════════════════════════════════════════════════════════════════════
# Problem: vLLM's FP8 activation quantization on Diffusion Transformer
# linear layers causes NaN cascades — the latent range shifts across
# denoising timesteps and compounds activation quantization rounding
# errors until the representation collapses to NaN/black output.
#
# Initial diagnosis suggested this only affected FFN/MLP layers (the
# image-stream MLPs). In practice the ATTENTION layers (attn1.to_qkv,
# attn2.to_q, etc.) ALSO produce NaN — diagnostic showed the first FFN
# call already receives NaN input, proving attention is the upstream
# culprit. The fix therefore applies to ALL DiT linear layers (both
# attention projections and FFN sublayers).
#
# Fix: Replace Fp8LinearMethod on all DiT linear layers with a subclass
# that keeps the FP8 weight storage (so checkpoint + memory cost are
# unchanged) but OVERRIDES `apply` to dequantize weights to BF16 and run
# a BF16 matmul with un-quantized activations. This eliminates the
# activation quantization that was producing NaN, without changing the
# memory footprint.
#
# Why this approach (vs UnquantizedLinearMethod)?
#   * UnquantizedLinearMethod would store weights as BF16 (~2× memory).
#     For a 14B model with ~150 linear sublayers, that pushes total VRAM
#     past the 24GB RTX 4090 limit and OOMs during model construction.
#   * Allocating unquantized weights on CPU breaks vLLM's dummy run
#     (mat1 on cuda, weights on cpu → device-mismatch RuntimeError).
#   * The weight-only FP8 path matches vLLM's VLLM_BATCH_INVARIANT branch
#     (see Fp8LinearMethod.apply) — proven safe for sensitive DiT layers.
#
# This patch runs in the WORKER process (where the model is built) because
# this file replaces pipeline_wan2_2.py which is imported before the
# transformer layers are created.
# ════════════════════════════════════════════════════════════════════════
import logging as _logging
_fp8_patch_logger = _logging.getLogger("fp8_ffn_patch")

try:
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config as _Fp8Config,
        Fp8LinearMethod as _Fp8LinearMethod,
    )
    from vllm.model_executor.layers.linear import LinearBase as _LinearBase
    import torch as _torch

    _orig_get_quant_method = _Fp8Config.get_quant_method

    class _Fp8WeightOnlyLinearMethod(_Fp8LinearMethod):
        """FP8 weight storage + BF16 matmul (no activation quantization).

        Identical to Fp8LinearMethod except `apply` always takes the
        BF16-dequant + F.linear path that Fp8LinearMethod.apply uses when
        VLLM_BATCH_INVARIANT=1 (per-tensor / per-channel, non-block-quant).
        This avoids the FP8 activation quantization that produces NaN in
        Diffusion Transformer linear layers across denoising timesteps.

        Note: our checkpoint is **direct-cast FP8** (values already in the
        normal NN range ±0.4, no per-tensor scaling). The `weight_scale`
        parameter created by Fp8LinearMethod.create_weights stays
        uninitialized (zero) because the checkpoint has no scale tensors —
        so we MUST NOT multiply by it. We just cast FP8 → BF16 directly.
        """

        def apply(self, layer, x, bias=None):
            # Direct-cast FP8: the stored FP8 values ARE the actual weights.
            # Cast to BF16 and run a normal linear. No weight_scale, no
            # activation quantization → no NaN cascade on DiT layers.
            weight_bf16 = layer.weight.to(_torch.bfloat16)
            return _torch.nn.functional.linear(x, weight_bf16.t(), bias)

    def _patched_get_quant_method(self, layer, prefix):
        # Replace the FP8 method on ALL DiT linear sublayers with the
        # weight-only variant. Initial diagnosis targeted only FFN/MLP
        # layers, but in practice the attention layers (attn1, attn2) also
        # produce NaN under FP8 activation quantization in Diffusion
        # Transformers — the first FFN call already receives NaN input,
        # proving attention is the upstream culprit.
        #
        # We match any transformer.blocks.* or transformer.vace_blocks.*
        # linear layer. Other layers (text encoder, VAE, norms) keep the
        # original FP8 method.
        if isinstance(layer, _LinearBase) and (
            "ffn" in prefix or "attn" in prefix
        ) and ("net_0" in prefix or "net_2" in prefix
               or "to_q" in prefix or "to_k" in prefix
               or "to_v" in prefix or "to_out" in prefix
               or "to_qkv" in prefix or "proj" in prefix):
            return _Fp8WeightOnlyLinearMethod(self)
        return _orig_get_quant_method(self, layer, prefix)

    _Fp8Config.get_quant_method = _patched_get_quant_method

    # WARNING level so it shows in logs regardless of log config
    _fp8_patch_logger.warning(
        "FP8 weight-only patch applied on DiT attn+ffn — FP8 storage + "
        "BF16 matmul (no activation quantization → no NaN)"
    )
except Exception as _e:
    _fp8_patch_logger.warning("Could not apply FP8 FFN patch: %s", _e)
    import traceback as _tb
    _fp8_patch_logger.warning(_tb.format_exc())

# ════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, cast

import PIL.Image
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import WanVACETransformer3DModel as WanTransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)
DEBUG_PERF = False
WAN_SAMPLE_SOLVER_CHOICES = {"unipc", "euler"}


def build_wan_scheduler(sample_solver: str, flow_shift: float) -> Any:
    if sample_solver == "unipc":
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )
    if sample_solver == "euler":
        return WanEulerScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

    raise ValueError(
        f"Unsupported Wan sample_solver: {sample_solver}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}"
    )


def resolve_wan_sample_solver(req: OmniDiffusionRequest, default: str = "unipc") -> str:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN_SAMPLE_SOLVER_CHOICES:
        raise ValueError(f"Invalid sample_solver={raw!r}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}")
    return sample_solver


def resolve_wan_flow_shift(req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> float:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}. flow_shift must be a float.") from exc


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None, prefix: str = ""
) -> WanTransformer3DModel:
    """Create WanTransformer3DModel from config dict."""
    kwargs: dict = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]
    if "vace_layers" in config:
        kwargs["vace_layers"] = config["vace_layers"]
    if "vace_in_channels" in config:
        kwargs["vace_in_channels"] = config["vace_in_channels"]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if prefix:
        kwargs["prefix"] = prefix

    return WanTransformer3DModel(**kwargs)


def get_wan22_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video
        custom_output = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            custom_output["video_fps_multiplier"] = multiplier
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "custom_output": custom_output,
        }

    return post_process_func


def get_wan22_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for Wan2.2: optionally load and resize input image for I2V mode."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                continue

            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"""Unsupported image format {raw_image.__class__}.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            # Calculate dimensions based on aspect ratio if not provided
            if request.sampling_params.height is None or request.sampling_params.width is None:
                # Default max area for 720P
                max_area = 720 * 1280
                aspect_ratio = image.height / image.width

                # Calculate dimensions maintaining aspect ratio
                mod_value = 16  # Must be divisible by 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize image to target dimensions
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22Pipeline(
    nn.Module, PipelineParallelMixin, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Read model_index.json to detect expand_timesteps mode (for TI2V-5B)
        self.expand_timesteps = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
            # Check if this is a two-stage model (MoE with transformer_2)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            # For remote models, download and read model_index.json
            try:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    # Check transformer_2 from model_index
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load based on boundary_ratio
        # boundary_ratio=1.0: only load transformer_2 (low-noise stage only)
        # boundary_ratio=0.0: only load transformer (high-noise stage only)
        # otherwise: load both transformers
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        if load_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )
        if load_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )

        # ``from_pretrained_with_prefetch`` re-prefetches and retries if the
        # cache is still half-written (the missing-shard ``OSError`` and the
        # default-``UMT5Config`` size-mismatch ``RuntimeError`` seen on multi
        # -worker HSDP / ring launches), instead of crashing the worker.
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        # Load text_encoder on CPU (intentionally, to save ~5.3 GiB VRAM)
        from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch as _fpwp
        from transformers import UMT5EncoderModel as _UMT5
        self.text_encoder = _fpwp(
            _UMT5.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        # Explicitly force to CPU regardless of CUDA device context
        if hasattr(self.text_encoder, 'cpu'):
            self.text_encoder = self.text_encoder.cpu()
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Initialize transformers with correct config (weights loaded via load_weights)
        if load_transformer:
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = self._create_transformer(transformer_config)
        else:
            self.transformer = None

        if load_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = self._create_transformer(transformer_2_config)
        else:
            self.transformer_2 = None

        # Store the active transformer config
        if load_transformer:
            self.transformer_config = self.transformer.config
        elif load_transformer_2:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                set_forward_context_denoise_step_idx(step_idx)

                # Select model based on timestep and boundary_ratio
                # High noise stage (t >= boundary_timestep): use transformer
                # Low noise stage (t < boundary_timestep): use transformer_2
                if boundary_timestep is not None and t < boundary_timestep:
                    # Low noise stage - always use guidance_high for this stage
                    current_guidance_scale = guidance_high
                    if self.transformer_2 is not None:
                        current_model = self.transformer_2
                    elif self.transformer is not None:
                        # Fallback to transformer if transformer_2 not loaded
                        current_model = self.transformer
                    else:
                        raise RuntimeError("No transformer available for low-noise stage")
                else:
                    # High noise stage - always use guidance_low for this stage
                    current_guidance_scale = guidance_low
                    if self.transformer is not None:
                        current_model = self.transformer
                    elif self.transformer_2 is not None:
                        # Fallback to transformer_2 if transformer not loaded
                        current_model = self.transformer_2
                    else:
                        raise RuntimeError("No transformer available for high-noise stage")

                if self.expand_timesteps and latent_condition is not None:
                    # I2V mode: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps per patch - use floor division to match patch embedding
                    patch_size = self.transformer_config.patch_size
                    patch_height = latents.shape[3] // patch_size[1]
                    patch_width = latents.shape[4] // patch_size[2]

                    # Create mask at patch resolution (same as hidden states sequence length)
                    patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                    patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]  # Ensure correct dimensions
                    temp_ts = (patch_mask[0][0] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # T2V mode: standard forward
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        if _is_rank_zero():
            logger.info("DEBUG denoised latents: min=%s max=%s mean=%s std=%s",
                        latents.min().item(), latents.max().item(),
                        latents.mean().item(), latents.std().item())
        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float | tuple[float, float] = 4.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Get parameters from request or arguments
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Wan2.2 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # Respect per-request guidance_scale when explicitly provided.
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        # record guidance for properties
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Prefer engine-configured boundary_ratio, but allow per-request fallback.
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio

        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")

        # validate shapes
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            # Fallback to text_encoder dtype if no transformer loaded
            dtype = self.text_encoder.dtype

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        if DEBUG_PERF:
            # Sync GPU before timing to ensure accurate measurements
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif guidance_low > 1.0 or guidance_high > 1.0:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are given and guidance > 1."
                )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()
        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning(
                    """Received a list of image. Only a single image is supported by this model."""
                    """Taking only the first image for now."""
                )
            raw_image = raw_image[0]
        if raw_image is None:
            image = None
        elif isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        latent_condition = None
        first_frame_mask = None

        if self.expand_timesteps and image is not None:
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            # Preprocess image
            if isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
        )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For I2V mode: blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()
        if output_type == "latent":
            output = latents
        else:
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
            if _is_rank_zero():
                logger.info("DEBUG VAE input latents: min=%s max=%s mean=%s std=%s",
                            latents.min().item(), latents.max().item(),
                            latents.mean().item(), latents.std().item())
            output = self.vae.decode(latents, return_dict=False)[0]

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, "
                    "StagesSum=%.2f ms, PipelineWall=%.2f ms, Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        return DiffusionOutput(
            output=output, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")


# ---------------------------------------------------------------------------
# DMD2-distilled variant
# ---------------------------------------------------------------------------


class WanT2VDMD2Pipeline(DMD2PipelineMixin, Wan22Pipeline):
    """Wan 2.x T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()

# ── Monkey-patch: hande text_encoder on CPU with GPU inputs ──
import functools as _ft

def _patch_te_forward():
    from transformers import UMT5EncoderModel as _UMT5
    if getattr(_UMT5, '_patched_fwd', False):
        return
    _orig_fwd = _UMT5.forward
    @_ft.wraps(_orig_fwd)
    def _patched_fwd(self, input_ids=None, attention_mask=None, **kwargs):
        p_dev = next(self.parameters()).device  # CPU
        i_dev = input_ids.device if input_ids is not None else (attention_mask.device if attention_mask is not None else None)
        if p_dev != i_dev and i_dev is not None:
            with torch.device(p_dev):
                result = _orig_fwd(self, input_ids=input_ids.to(p_dev) if input_ids is not None else None,
                                   attention_mask=attention_mask.to(p_dev) if attention_mask is not None else None,
                                   **kwargs)
                if hasattr(result, "last_hidden_state"):
                    result.last_hidden_state = result.last_hidden_state.to(i_dev)
                return result
        return _orig_fwd(self, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    _UMT5.forward = _patched_fwd
    _UMT5._patched_fwd = True

_patch_te_forward()

# ════════════════════════════════════════════════════════════════════════
# TEACACHE PATCH (Timestep Embedding Aware Cache)
# ════════════════════════════════════════════════════════════════════════
# Toggle: set OMNI_TEACACHE_THRESH=0 (off) or 0.001-0.1 (on).
#
# Principle: Consecutive diffusion timesteps produce similar time
# embeddings, so the transformer block outputs change slowly. When
# the timestep_proj L1 distance between consecutive steps is below a
# threshold, we reuse the cached block output instead of recomputing.
#
# Cache signal: timestep_proj (second output of condition_embedder).
# Distance metric: mean L1 over all feature dimensions, NO polynomial
# rescaling (the official coefficients are calibrated for Wan2.1 T2V
# 14B but produce abs(poly) ≈ 5-13 for typical distances of 0.001-0.04,
# making the threshold never fire — so we use raw distance directly).
#
# Recommended thresholds (raw L1, no polynomial):
#   0.001-0.003  conservative (~10-20% steps cached)
#   0.005-0.010  balanced (~30-50% steps cached, good quality)
#   0.015-0.030  aggressive (~50-70% steps cached, some quality loss)
#   0.050+       very aggressive (quality may degrade noticeably)
#
# VACE handling: vace_blocks always run unconditionally. Only the main
# self.blocks DiT loop is subject to caching.
#
# CFG handling: even-indexed forward passes (cond) and odd-indexed
# (uncond) maintain independent cache state. First 2 CFG pairs always
# compute to seed the cache.
#
# State machine (per CFG branch):
#   self._tc_state[counter % 2] = {
#       "prev_timestep_proj": tensor or None,
#       "cached_hidden_states": tensor or None,
#       "retention_left": 2,  # forced-compute countdown
#   }
# ════════════════════════════════════════════════════════════════════════
import os as _tc_os

_tc_thresh = float(_tc_os.environ.get("OMNI_TEACACHE_THRESH", "0"))

if _tc_thresh > 0:
    _tc_logger = _logging.getLogger("teacache_patch")

    try:
        _WanCls = WanTransformer3DModel
        _orig_wan_forward = _WanCls.forward

        def _teacache_forward(self, hidden_states, timestep, encoder_hidden_states,
                              encoder_hidden_states_image=None, return_dict=True,
                              attention_kwargs=None, vace_context=None,
                              vace_context_scale=1.0):
            """Forward with TeaCache — cached DiT blocks loop.

            Replicates WanVACETransformer3DModel.forward exactly except the
            main blocks loop (self.blocks) is cached via timestep_proj L1
            distance threshold.
            """
            batch_size, _, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            # ── RoPE ───────────────────────────────────────────────────
            current_rope_resolution = (post_patch_num_frames, post_patch_height, post_patch_width)
            if self._cached_rope_resolution == current_rope_resolution and self._cached_rope_emb is not None:
                rotary_emb = self._cached_rope_emb
            else:
                freqs_cos, freqs_sin = self.rope(hidden_states)
                rotary_emb = (freqs_cos[..., 0::2].to(hidden_states.dtype),
                              freqs_sin[..., 1::2].to(hidden_states.dtype))
                self._hidden_states_shape = hidden_states.shape
                self._cached_rope_emb = rotary_emb

            # ── Patch embedding ────────────────────────────────────────
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            # ── Timestep handling ──────────────────────────────────────
            if timestep.ndim == 2:
                ts_seq_len = timestep.shape[1]
                timestep = timestep.flatten()
            else:
                ts_seq_len = None

            # ── Condition embedding → temb + timestep_proj (cache signal) ──
            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = \
                self.condition_embedder(
                    timestep, encoder_hidden_states, encoder_hidden_states_image,
                    timestep_seq_len=ts_seq_len,
                )
            timestep_proj = self.timestep_proj_prepare(timestep_proj, ts_seq_len)

            if encoder_hidden_states_image is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1
                )

            # ── SP shard point ─────────────────────────────────────────
            hidden_states = self._sp_shard_point(hidden_states)

            # ── SP mask ────────────────────────────────────────────────
            hidden_states_mask = None
            ctx = get_forward_context()
            parallel_config = ctx.omni_diffusion_config.parallel_config
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    batch_size, padded_seq_len, dtype=torch.bool, device=hidden_states.device,
                )
                hidden_states_mask[:, ctx.sp_original_seq_len:] = False

            # ── VACE blocks (always run, never cached) ─────────────────
            vace_hints = None
            if vace_context is not None and self.vace_blocks is not None:
                sp_size = parallel_config.sequence_parallel_size if parallel_config is not None else 1
                full_seq_len = hidden_states.shape[1] * sp_size
                control_hidden_states = self.embed_vace_context(
                    vace_context.to(hidden_states.dtype), full_seq_len, sp_size,
                )
                vace_hints = []
                for block in self.vace_blocks:
                    conditioning_states, control_hidden_states = block(
                        hidden_states, encoder_hidden_states, control_hidden_states,
                        timestep_proj, rotary_emb, hidden_states_mask,
                    )
                    vace_hints.append(conditioning_states)

            # Normalize scale to per-layer list
            if vace_hints is not None and isinstance(vace_context_scale, (int, float)):
                vace_context_scale = [vace_context_scale] * len(vace_hints)

            # ── TeaCache state management ──────────────────────────────
            if not hasattr(self, '_tc_call_counter'):
                self._tc_call_counter = 0
            branch_key = self._tc_call_counter % 2  # 0=cond, 1=uncond
            self._tc_call_counter += 1

            if not hasattr(self, '_tc_state'):
                self._tc_state = {}
            if branch_key not in self._tc_state:
                self._tc_state[branch_key] = {
                    "prev_timestep_proj": None,
                    "cached_hidden_states": None,
                    "retention_left": 2,
                }
            tc_s = self._tc_state[branch_key]

            # ── Cache decision (raw L1 distance, no polynomial) ────────
            current_proj = timestep_proj.detach().float()
            use_cache = False
            if tc_s["prev_timestep_proj"] is not None and tc_s["cached_hidden_states"] is not None:
                # Mean L1 distance between current and previous timestep_proj
                l1_dist = torch.nn.functional.l1_loss(
                    current_proj, tc_s["prev_timestep_proj"], reduction='mean'
                ).item()
                use_cache = (l1_dist < _tc_thresh and tc_s["retention_left"] <= 0)
            # retention_left applies even on first call (prev is None)
            if tc_s["retention_left"] > 0:
                tc_s["retention_left"] -= 1

            if use_cache:
                # ── Reuse cached hidden states ─────────────────────────
                hidden_states = tc_s["cached_hidden_states"]
            else:
                # ── Full compute through all DiT blocks ────────────────
                for i, block in enumerate(self.blocks):
                    hidden_states = block(
                        hidden_states, encoder_hidden_states, timestep_proj,
                        rotary_emb, hidden_states_mask,
                    )
                    if vace_hints is not None and self.vace_layers_mapping is not None \
                            and i in self.vace_layers_mapping:
                        vace_idx = self.vace_layers_mapping[i]
                        hidden_states = hidden_states + \
                            vace_hints[vace_idx] * vace_context_scale[vace_idx]

                tc_s["cached_hidden_states"] = hidden_states.clone()

            tc_s["prev_timestep_proj"] = current_proj

            # ── Output norm, projection & unpatchify ───────────────────
            shift, scale = self.output_scale_shift_prepare(temb)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            if shift.ndim == 2:
                shift = shift.unsqueeze(1)
                scale = scale.unsqueeze(1)

            hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)

            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height,
                post_patch_width, p_t, p_h, p_w, -1,
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        _WanCls.forward = _teacache_forward
        _tc_logger.warning(
            "TeaCache ENABLED (raw L1 thresh=%.4f) — Wan VACE 14B "
            "blocks loop cached via timestep_proj signal. "
            "Set OMNI_TEACACHE_THRESH=0 to disable.",
            _tc_thresh,
        )
    except Exception as _tc_e:
        _tc_logger.warning("Could not apply TeaCache patch: %s", _tc_e)
        import traceback as _tc_tb
        _tc_logger.warning(_tc_tb.format_exc())
else:
    pass  # TeaCache disabled (OMNI_TEACACHE_THRESH=0 or unset)
