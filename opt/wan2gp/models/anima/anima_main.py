"""Anima model factory — Cosmos-Predict2-2B via diffusers Cosmos2TextToImagePipeline.

Loads the Cosmos transformer, Qwen3 0.6B text encoder, Qwen-Image VAE,
and constructs a standard Cosmos2TextToImagePipeline from diffusers.
"""
import json
import os
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler, Cosmos2TextToImagePipeline
from diffusers import CosmosTransformer3DModel
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, Qwen3ForCausalLM

# Reuse Z-Image's AutoencoderKL — same VAE checkpoint, same interface.
# Cosmos2TextToImagePipeline just calls vae.encode()/decode(), doesn't
# care about the specific VAE class. Avoids config incompatibility with
# diffusers' AutoencoderKLCosmos.
import sys as _sys
_zimg_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "z_image"
)
if _zimg_dir not in _sys.path:
    _sys.path.insert(0, _zimg_dir)
from autoencoder_kl import AutoencoderKL

logger = logging.get_logger(__name__)


def _strip_net_prefix(sd: dict) -> dict:
    """Strip 'net.' prefix from ComfyUI-serialized Cosmos state dict keys.

    ComfyUI wraps the transformer in a 'net' module during save.
    diffusers' CosmosTransformer3DModel expects keys without the prefix.
    """
    out = {}
    for key, tensor in sd.items():
        if key.startswith("net."):
            out[key[4:]] = tensor
        else:
            out[key] = tensor
    return out


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        **kwargs,
    ):
        model_def = model_def or {}
        self.base_model_type = base_model_type
        self.model_def = model_def

        # --- Transformer ---
        transformer_filename = (
            model_filename[0] if isinstance(model_filename, (list, tuple))
            else model_filename
        )
        if transformer_filename is None:
            raise ValueError("No transformer filename provided for Anima.")

        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs", f"{base_model_type}.json"
        )

        with open(config_path, "r") as f:
            config = json.load(f)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        # Load transformer with config, stripping net. prefix from weights
        with init_empty_weights():
            transformer = CosmosTransformer3DModel(**config)

        offload.load_model_data(
            transformer,
            model_filename,
            writable_tensors=False,
            preprocess_sd=_strip_net_prefix,
        )
        transformer.to(dtype)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(
                transformer, model_type, transformer_filename, dtype, config_path
            )

        # --- Text encoder ---
        text_encoder = offload.fast_load_transformers_model(
            text_encoder_filename,
            writable_tensors=True,
            modelClass=Qwen3ForCausalLM,
        )

        # --- Tokenizer ---
        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = os.path.dirname(
                fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
            )
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # --- VAE (Qwen-Image VAE, same as Z-Image) ---
        vae_filename = fl.locate_file("ZImageTurbo_VAE_bf16.safetensors")
        vae_config_path = fl.locate_file("ZImageTurbo_VAE_bf16_config.json")

        vae = offload.fast_load_transformers_model(
            vae_filename,
            writable_tensors=True,
            modelClass=AutoencoderKL,
            defaultConfigPath=vae_config_path,
            default_dtype=VAE_dtype,
        )

        # --- Scheduler ---
        with open(fl.locate_file("ZImageTurbo_scheduler_config.json"), "r") as f:
            scheduler_config = json.load(f)
        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        # --- Pipeline ---
        self.pipeline = Cosmos2TextToImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 30,
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 4.0,
        batch_size: int = 1,
        callback=None,
        max_sequence_length: int = 512,
        VAE_tile_size=None,
        loras_slists=None,
        **kwargs,
    ):
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if seed is None or seed < 0:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling = VAE_tile_size > 0
                tile_size = max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            self.vae.use_tiling = tiling
            self.vae.tile_latent_min_height = tile_size
            self.vae.tile_latent_min_width = tile_size

        images = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt or "worst quality, low quality",
            num_inference_steps=sampling_steps,
            guidance_scale=guide_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=None,
            output_type="pt",
            return_dict=True,
        )

        if images is None:
            return None

        if not torch.is_tensor(images):
            images = torch.tensor(images)

        return images.transpose(0, 1)

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
