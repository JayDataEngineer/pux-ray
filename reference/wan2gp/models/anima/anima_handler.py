"""Anima family handler — Cosmos-Predict2-2B text-to-image.

Anima is a 2B anime/illustration model by CircleStone Labs + Comfy Org,
built on NVIDIA Cosmos architecture. Uses Qwen3 0.6B text encoder and
Qwen-Image VAE (same VAE as the Qwen/Z-Image family, loaded via
AutoencoderKLQwenImage in diffusers format).

Files downloaded:
  - Transformer: from defaults/anima_base.json URL (ComfyUI format, key-converted)
  - Text encoder: circlestone-labs/Anima → split_files/text_encoders/qwen_3_06b_base.safetensors
  - VAE: DeepBeepMeep/Qwen_image → qwen_vae.safetensors (diffusers format)
  - VAE config: DeepBeepMeep/Qwen_image → qwen_vae_config.json
"""
import os
import torch
from shared.utils.hf import build_hf_url


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
        }
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("circlestone-labs/Anima", "split_files/text_encoders", "qwen_3_06b_base.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = "split_files/text_encoders"
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["anima_base"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "anima"

    @staticmethod
    def query_family_infos():
        return {"anima": (130, "Anima")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-anima",
            type=str,
            default=None,
            help=f"Path to a directory that contains anima Loras (default: {os.path.join(lora_root, 'anima')})"
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_anima", None) or os.path.join(lora_root, "anima")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        download_def = [
            {
                # Text encoder from Anima repo (ComfyUI format)
                "repoId": "circlestone-labs/Anima",
                "sourceFolderList": ["split_files/text_encoders"],
                "fileList": [
                    ["qwen_3_06b_base.safetensors"],
                ],
            },
            {
                # VAE in diffusers format from Qwen Image repo — the ComfyUI-format
                # VAE from circlestone-labs/Anima has different state_dict keys that
                # don't match AutoencoderKLCosmos OR AutoencoderKLQwenImage (0/194
                # keys matched).  The diffusers-format file has matching keys.
                "repoId": "DeepBeepMeep/Qwen_image",
                "sourceFolderList": [""],
                "fileList": [
                    ["qwen_vae.safetensors", "qwen_vae_config.json"],
                ],
            },
        ]
        return download_def

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .anima_main import model_factory
        from shared.utils import files_locator as _fl

        pipe_processor = model_factory(
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae,
        }
        return pipe_processor, pipe

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale": 4.5,
            "num_inference_steps": 30,
            "flow_shift": 3.0,
            "n_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
        })
