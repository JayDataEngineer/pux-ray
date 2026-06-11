"""Anima family handler — Minimal implementation to avoid recursion."""
import os
import torch

# Removed: from shared.utils.hf import build_hf_url (potential recursion source)

class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        # Minimal model def to avoid recursion
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
            "text_encoder_folder": "Qwen3-0.6B",
        }
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
        pass  # Skip for now

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, "anima")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        # Skip download definitions for now
        return []

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

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
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
        # Return dummy factors for now
        return None, None

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale": 4,
            "num_inference_steps": 30,
            "flow_shift": 6.0,
        })
