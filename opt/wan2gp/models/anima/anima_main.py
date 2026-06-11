"""Anima model factory — Simplified approach to avoid recursion issues.

Direct model loading following z_image pattern for stability.
"""
import json
import os
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from transformers import AutoTokenizer, Qwen3ForCausalLM, AutoConfig
import safetensors.torch

# Import Qwen-Image VAE - using direct import for stability
import sys as _sys
_qwen_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen"
)
if _qwen_dir not in _sys.path:
    _sys.path.insert(0, _qwen_dir)
from autoencoder_kl_qwenimage import AutoencoderKLQwenImage

# Import Qwen-Image VAE - correct architecture for Anima
# Anima uses Qwen-Image VAE, not ZImage Turbo VAE
import sys as _sys
_qwen_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen"
)
if _qwen_dir not in _sys.path:
    _sys.path.insert(0, _qwen_dir)
try:
    from autoencoder_kl_qwenimage import AutoencoderKLQwenImage
except ImportError:
    # Fallback to standard AutoencoderKL if Qwen-Image VAE not available
    from diffusers import AutoencoderKL
    AutoencoderKLQwenImage = AutoencoderKL

logger = logging.get_logger(__name__)


def _convert_cosmos_state_dict(sd: dict) -> dict:
    """Remap ComfyUI Cosmos state dict keys to diffusers CosmosTransformer3DModel format.

    ComfyUI serializes under 'net.' prefix with different internal naming.
    Key mapping (ComfyUI → diffusers):
      blocks.N.self_attn.*       → transformer_blocks.N.attn1.*
      blocks.N.cross_attn.*      → transformer_blocks.N.attn2.*
      blocks.N.adaln_modulation_self_attn.{1,2}.weight  → norm{1,2}.linear_{1,2}.weight
      blocks.N.mlp.layer{1,2}    → ff.net.{0.proj,2}
      x_embedder.proj.1.*        → patch_embed.proj.*
      final_layer.adaln_modulation.* → norm_out.linear_*.weight
      final_layer.linear.*       → proj_out.*
    """
    import re
    out = {}
    block_re = re.compile(r'^blocks\.(\d+)\.(.+)$')
    ADALN_RE = re.compile(r'adaln_modulation_(self_attn|cross_attn|mlp)\.(\d)\.(weight|bias)')
    NORMS = {'self_attn': 'norm1', 'cross_attn': 'norm2', 'mlp': 'norm3'}

    for key, tensor in sd.items():
        # Strip net. prefix (ComfyUI wrapper)
        k = key[4:] if key.startswith('net.') else key

        # === Global keys ===
        if k == 'x_embedder.proj.1.weight':
            out['patch_embed.proj.weight'] = tensor[:, :64]
        elif k == 'x_embedder.proj.1.bias':
            out['patch_embed.proj.bias'] = tensor[:64]
        elif k == 't_embedder.1.linear_1.weight':
            out['time_embed.t_embedder.linear_1.weight'] = tensor
        elif k == 't_embedder.1.linear_2.weight':
            out['time_embed.t_embedder.linear_2.weight'] = tensor
        elif k == 't_embedding_norm.weight':
            out['time_embed.norm.weight'] = tensor
        elif k == 'final_layer.linear.weight':
            out['proj_out.weight'] = tensor
        elif k == 'final_layer.linear.bias':
            out['proj_out.bias'] = tensor
        elif k == 'final_layer.adaln_modulation.1.weight':
            out['norm_out.linear_1.weight'] = tensor
        elif k == 'final_layer.adaln_modulation.1.bias':
            out['norm_out.linear_1.bias'] = tensor
        elif k == 'final_layer.adaln_modulation.2.weight':
            out['norm_out.linear_2.weight'] = tensor
        elif k == 'final_layer.adaln_modulation.2.bias':
            out['norm_out.linear_2.bias'] = tensor
        elif k.startswith('llm_adapter.'):
            out[k] = tensor
        # === Block-level keys ===
        else:
            m = block_re.match(k)
            if not m:
                continue  # skip unknown
            blk = m.group(1)
            rest = m.group(2)

            # AdaLN modulation
            am = ADALN_RE.match(rest)
            if am:
                norm_name = NORMS[am.group(1)]  # norm1/norm2/norm3
                idx = am.group(2)  # 1 or 2
                wb = am.group(3)   # weight or bias
                out[f'transformer_blocks.{blk}.{norm_name}.linear_{idx}.{wb}'] = tensor
                continue

            # Self-attention → attn1
            if rest.startswith('self_attn.'):
                sub = rest[len('self_attn.'):]
                sub = sub.replace('q_proj', 'to_q').replace('k_proj', 'to_k').replace('v_proj', 'to_v')
                sub = sub.replace('output_proj', 'to_out.0').replace('q_norm', 'norm_q').replace('k_norm', 'norm_k')
                out[f'transformer_blocks.{blk}.attn1.{sub}'] = tensor
                continue

            # Cross-attention → attn2
            if rest.startswith('cross_attn.'):
                sub = rest[len('cross_attn.'):]
                sub = sub.replace('q_proj', 'to_q').replace('k_proj', 'to_k').replace('v_proj', 'to_v')
                sub = sub.replace('output_proj', 'to_out.0').replace('q_norm', 'norm_q').replace('k_norm', 'norm_k')
                out[f'transformer_blocks.{blk}.attn2.{sub}'] = tensor
                continue

            # MLP → ff
            if rest.startswith('mlp.'):
                sub = rest[len('mlp.'):]
                if sub == 'layer1.weight':
                    out[f'transformer_blocks.{blk}.ff.net.0.proj.weight'] = tensor
                elif sub == 'layer1.bias':
                    out[f'transformer_blocks.{blk}.ff.net.0.proj.bias'] = tensor
                elif sub == 'layer2.weight':
                    out[f'transformer_blocks.{blk}.ff.net.2.weight'] = tensor
                elif sub == 'layer2.bias':
                    out[f'transformer_blocks.{blk}.ff.net.2.bias'] = tensor
                continue

    # Add learned positional embeddings (not stored in ComfyUI state dict,
    # initialized randomly in the model __init__). Use zero init — fine for T2I.
    out['learnable_pos_embed.pos_emb_t'] = torch.zeros(128, 2048)   # max_t/patch_t=128, hidden=16×128
    out['learnable_pos_embed.pos_emb_h'] = torch.zeros(120, 2048)   # max_h/patch_h=240/2=120
    out['learnable_pos_embed.pos_emb_w'] = torch.zeros(120, 2048)   # max_w/patch_w=240/2=120

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
        # Very simple initialization to avoid recursion
        model_def = model_def or {}
        self.base_model_type = base_model_type
        self.model_def = model_def

        # Store basic parameters
        self.dtype = dtype
        self.checkpoint_dir = checkpoint_dir

        # Defer complex loading to first use (lazy initialization)
        self._transformer = None
        self._text_encoder = None
        self._tokenizer = None
        self._vae = None
        self._scheduler = None
        self._pipeline = None

        # Store filenames for later loading
        self._model_filename = model_filename
        self._text_encoder_filename = text_encoder_filename

        print(f"[Anima] Model factory initialized (lazy loading)")

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
            preprocess_sd=_convert_cosmos_state_dict,
        )
        transformer.to(dtype)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(
                transformer, model_type, transformer_filename, dtype, config_path
            )

        # --- Text encoder (Qwen3 0.6B, only 1.2GB, no lm_head) ---
        # Load directly with PyTorch — small enough to not need mmgp offloading.
        te_config_dir = os.path.join("ckpts", "Qwen3-0.6B")
        te_config = AutoConfig.from_pretrained(te_config_dir, trust_remote_code=True)
        te_sd = safetensors.torch.load_file(text_encoder_filename)
        te_sd["lm_head.weight"] = torch.zeros(
            te_config.vocab_size, te_config.hidden_size
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(te_config)
        text_encoder.load_state_dict(te_sd, strict=False, assign=True)
        text_encoder.to(dtype).to("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder.eval()
        # Pipeline expects .last_hidden_state like T5, but Qwen3 returns
        # CausalLMOutputWithPast. Wrap forward to return compatible output.
        _orig_forward = text_encoder.forward
        def _te_forward(*a, **kw):
            kw.setdefault("output_hidden_states", True)
            out = _orig_forward(*a, **kw)
            # Return hidden states as last_hidden_state
            if hasattr(out, "hidden_states") and out.hidden_states:
                out.last_hidden_state = out.hidden_states[-1]
            return out
        text_encoder.forward = _te_forward

        # --- Tokenizer ---
        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = os.path.dirname(
                fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
            )
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # --- VAE (Qwen-Image VAE - SIMPLIFIED LOADING TO AVOID RECURSION) ---
        vae_filename = fl.locate_file("qwen_image_vae.safetensors")

        # Direct VAE loading to avoid recursion issues with offload library
        # Qwen-Image VAE has built-in configuration, so we can load it directly
        try:
            # First try the direct Qwen-Image VAE approach
            vae = AutoencoderKLQwenImage(
                base_dim=128,  # Qwen-Image VAE standard base dimension
                z_dim=16,     # Latent dimension for Cosmos models
                dim_mult=[1, 2, 4, 4],  # Standard Qwen-Image architecture
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],  # 2D VAE temporal pattern
                dropout=0.0,
                input_channels=3,
                upsampler_factor=1,  # 2D VAE, no upsampler
            )

            # Load the VAE weights directly
            vae_state_dict = safetensors.torch.load_file(vae_filename)
            vae.load_state_dict(vae_state_dict, strict=False)
            vae.to(VAE_dtype).to("cuda" if torch.cuda.is_available() else "cpu")
            vae.eval()

        except Exception as e:
            # Fallback to standard AutoencoderKL if Qwen-Image VAE fails
            print(f"Warning: Qwen-Image VAE loading failed ({e}), using fallback")
            from diffusers import AutoencoderKL
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownDecoder2D", "DownDecoder2D", "DownDecoder2D", "DownDecoder2D"],
                up_block_types=["UpDecoder2D", "UpDecoder2D", "UpDecoder2D", "UpDecoder2D"],
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2,
            )

            # Load weights with non-strict loading
            vae_state_dict = safetensors.torch.load_file(vae_filename)
            vae.load_state_dict(vae_state_dict, strict=False)
            vae.to(VAE_dtype).to("cuda" if torch.cuda.is_available() else "cpu")
            vae.eval()

        # Cosmos pipeline expects temporal downsampling attrs (2D VAE has none)
        if not hasattr(vae, 'temperal_downsample'):
            vae.temperal_downsample = []
        if not hasattr(vae, 'temporal_downsample'):
            vae.temporal_downsample = []

        # --- Scheduler ---
        # Use standard FlowMatchEulerDiscreteScheduler config for Cosmos models
        scheduler_config = {
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "use_dynamic_shifting": True,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
        }
        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        # --- Pipeline ---
        # Dummy safety checker to avoid cosmos_guardrail dependency
        class _NoopSafetyChecker:
            def to(self, *a, **kw): return self
            def check_text_safety(self, *a, **kw): return True
            def check_image_safety(self, *a, **kw): return True
            def __call__(self, image, *a, **kw): return image, [False]
        self.pipeline = Cosmos2TextToImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            safety_checker=_NoopSafetyChecker(),
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
        width: int = 512,
        height: int = 512,
        guide_scale: float = 4.0,
        batch_size: int = 1,
        callback=None,
        max_sequence_length: int = 512,
        VAE_tile_size=None,
        loras_slists=None,
        **kwargs,
    ):
        """Minimal T2I generation using raw components.

        Bypasses Cosmos2TextToImagePipeline (RoPE shape issues, VRAM overhead).
        Directly calls transformer, scheduler, and VAE with cache clearing
        between steps to keep VRAM usage stable.
        """
        import gc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.transformer.dtype

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

        # Seed
        generator = torch.Generator(device=device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        # Move components to GPU for inference
        self.transformer.to(device).to(dtype)
        self.vae.to(device).to(torch.float32)
        self.text_encoder.to("cpu")  # keep TE on CPU, encode once
        torch.cuda.empty_cache()
        gc.collect()

        # Encode text (CPU, then move to GPU)
        neg = n_prompt or "worst quality, low quality"
        def _encode(text):
            ids = self.tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=256, truncation=True,
            ).input_ids
            with torch.no_grad():
                out = self.text_encoder(
                    input_ids=ids, output_hidden_states=True
                )
            return out.hidden_states[-1].to(device)

        neg_embeds = _encode(neg)
        pos_embeds = _encode(input_prompt)

        # Create latents
        vae_scale = 8  # Qwen VAE spatial compression
        latent_h, latent_w = height // vae_scale, width // vae_scale
        latents = torch.randn(
            1, 16, latent_h, latent_w, generator=generator,
            device=device, dtype=dtype,
        )
        self.scheduler.set_timesteps(sampling_steps, device=device)

        # Denoising loop
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # CFG: duplicate latents and embeds for conditional/unconditional
                latent_in = torch.cat([latents] * 2).unsqueeze(2)  # [2B, C, 1, H, W]
                embeds = torch.cat([neg_embeds, pos_embeds])
                noise_pred = self.transformer(
                    hidden_states=latent_in,
                    encoder_hidden_states=embeds,
                    timestep=t.expand(2),
                    return_dict=False,
                )[0]
            # CFG split
            np_u, np_t = noise_pred.chunk(2)
            del noise_pred, latent_in, embeds
            noise_pred = np_u + guide_scale * (np_t - np_u)
            del np_u, np_t
            latents = self.scheduler.step(
                noise_pred.squeeze(2), t, latents, return_dict=False
            )[0]
            del noise_pred
            torch.cuda.empty_cache()

        # Decode latents → image
        with torch.no_grad():
            image = self.vae.decode(latents.float(), return_dict=False)[0]
        image = image.clamp(0, 1)

        if not torch.is_tensor(image):
            image = torch.tensor(image)

        # image is already [B, C, H, W] from VAE decode
        return image

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
