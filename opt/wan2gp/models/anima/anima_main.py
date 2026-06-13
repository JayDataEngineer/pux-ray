"""Anima model factory — Cosmos-Predict2-2B text-to-image.

Loads the Cosmos transformer, Qwen3 0.6B text encoder, Qwen-Image VAE
(16 latent channels via AutoencoderKLQwenImage), and builds a minimal
generation pipeline that includes the LLM adapter for text conditioning.

Architecture flow (matching ComfyUI's Anima node):
  1. Tokenize text with Qwen3 tokenizer → Qwen3 token IDs
  2. Tokenize text with T5 tokenizer → T5 token IDs
  3. Qwen3 text encoder → hidden states [B, seq, 1024]
  4. LLM adapter: cross-attend T5 embeddings to Qwen3 hidden states
     → transformed embeddings [B, t5_seq, 1024]
  5. Pad to 512 tokens
  6. Cosmos transformer: denoise latents conditioned on transformed embeddings
  7. VAE decode: latents → image
"""
import gc
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler, CosmosTransformer3DModel
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, AutoConfig, Qwen3ForCausalLM

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm — ComfyUI's operations.RMSNorm equivalent.  LayerNorm would
# silently load weights but compute a different normalisation.
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# LLM Adapter — mirrors ComfyUI's comfy/ldm/anima/model.py exactly:
#   LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024,
#              num_layers=6, use_self_attn=True, layer_norm=False)
#
# The adapter takes:
#   source_hidden_states: Qwen3 hidden states [B, src_len, 1024]
#   target_input_ids:     T5 token IDs [B, tgt_len]
# And returns transformed embeddings [B, tgt_len, 1024].
#
# Key structure must match the checkpoint's llm_adapter.* keys exactly.
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class _AdapterAttention(nn.Module):
    """Attention matching ComfyUI's anima Attention (RMSNorm q/k norms, o_proj)."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = _RMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = _RMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        B, S, _ = x.shape
        _, C, _ = context.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(B, C, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _apply_rotary_pos_emb(q, cos, sin)
            cos, sin = position_embeddings_context
            k = _apply_rotary_pos_emb(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1).contiguous())


class _AdapterBlock(nn.Module):
    """TransformerBlock from ComfyUI anima (use_self_attn=True, layer_norm=False → RMSNorm)."""

    def __init__(self, source_dim=1024, model_dim=1024, num_heads=16):
        super().__init__()
        head_dim = model_dim // num_heads
        self.norm_self_attn = _RMSNorm(model_dim)
        self.self_attn = _AdapterAttention(model_dim, model_dim, num_heads, head_dim)
        self.norm_cross_attn = _RMSNorm(model_dim)
        self.cross_attn = _AdapterAttention(model_dim, source_dim, num_heads, head_dim)
        self.norm_mlp = _RMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim, bias=True),
        )

    def forward(self, x, context, pos_emb=None, pos_emb_ctx=None):
        # Self-attention with RoPE
        attn_out = self.self_attn(
            self.norm_self_attn(x),
            position_embeddings=pos_emb,
            position_embeddings_context=pos_emb,
        )
        x = x + attn_out
        # Cross-attention with RoPE
        attn_out = self.cross_attn(
            self.norm_cross_attn(x),
            context=context,
            position_embeddings=pos_emb,
            position_embeddings_context=pos_emb_ctx,
        )
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        return x


class _LLMAdapter(nn.Module):
    """Exact replica of ComfyUI's LLMAdapter for weight compatibility."""

    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024,
                 num_layers=6, num_heads=16):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        self.in_proj = nn.Identity()  # model_dim == target_dim
        self.rotary_emb = _RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            _AdapterBlock(source_dim, model_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim, bias=True)
        self.norm = _RMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids):
        """Transform text embeddings via cross-attention.

        Args:
            source_hidden_states: [B, src_len, source_dim] — Qwen3 hidden states
            target_input_ids: [B, tgt_len] or [tgt_len] — T5 token IDs

        Returns:
            [B, tgt_len, target_dim] — transformed embeddings
        """
        if target_input_ids.ndim == 1:
            target_input_ids = target_input_ids.unsqueeze(0)

        context = source_hidden_states
        x = self.embed(target_input_ids).to(context.dtype)

        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_ids_ctx = torch.arange(context.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary_emb(x, pos_ids)
        pos_emb_ctx = self.rotary_emb(x, pos_ids_ctx)

        for block in self.blocks:
            x = block(x, context, pos_emb=pos_emb, pos_emb_ctx=pos_emb_ctx)

        return self.norm(self.out_proj(x))


def _attach_llm_adapter(transformer, text_embed_dim=1024):
    """Build and attach the LLM adapter as a named submodule of the transformer."""
    transformer.llm_adapter = _LLMAdapter(source_dim=text_embed_dim)


# ---------------------------------------------------------------------------
# State-dict conversion: ComfyUI keys → diffusers CosmosTransformer3DModel keys
# ---------------------------------------------------------------------------
_ADALN_RE = re.compile(r'adaln_modulation_(self_attn|cross_attn|mlp)\.(\d)\.(weight|bias)')
_NORMS = {'self_attn': 'norm1', 'cross_attn': 'norm2', 'mlp': 'norm3'}
_BLOCK_RE = re.compile(r'^blocks\.(\d+)\.(.+)$')


def _convert_cosmos_state_dict(sd: dict) -> dict:
    out = {}
    for key, tensor in sd.items():
        k = key[4:] if key.startswith('net.') else key

        # --- Global keys ---
        if k == 'x_embedder.proj.1.weight':
            # ComfyUI: Linear [model_ch, 64]; diffusers: same shape [model_ch, 64]
            out['patch_embed.proj.weight'] = tensor
        elif k == 'x_embedder.proj.1.bias':
            out['patch_embed.proj.bias'] = tensor
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
        # LLM adapter — keep as-is
        elif k.startswith('llm_adapter.'):
            out[k] = tensor
        # --- Block-level keys ---
        else:
            m = _BLOCK_RE.match(k)
            if not m:
                continue
            blk, rest = m.group(1), m.group(2)

            # AdaLN modulation
            am = _ADALN_RE.match(rest)
            if am:
                norm_name = _NORMS[am.group(1)]
                idx, wb = am.group(2), am.group(3)
                out[f'transformer_blocks.{blk}.{norm_name}.linear_{idx}.{wb}'] = tensor
                continue

            # Self-attention → attn1
            if rest.startswith('self_attn.'):
                sub = rest[len('self_attn.'):]
                sub = sub.replace('q_proj', 'to_q').replace('k_proj', 'to_k')
                sub = sub.replace('v_proj', 'to_v').replace('output_proj', 'to_out.0')
                sub = sub.replace('q_norm', 'norm_q').replace('k_norm', 'norm_k')
                out[f'transformer_blocks.{blk}.attn1.{sub}'] = tensor
                continue

            # Cross-attention → attn2
            if rest.startswith('cross_attn.'):
                sub = rest[len('cross_attn.'):]
                sub = sub.replace('q_proj', 'to_q').replace('k_proj', 'to_k')
                sub = sub.replace('v_proj', 'to_v').replace('output_proj', 'to_out.0')
                sub = sub.replace('q_norm', 'norm_q').replace('k_norm', 'norm_k')
                out[f'transformer_blocks.{blk}.attn2.{sub}'] = tensor
                continue

            # MLP → ff
            if rest.startswith('mlp.'):
                sub = rest[len('mlp.'):]
                mapping = {
                    'layer1.weight': 'ff.net.0.proj.weight',
                    'layer1.bias': 'ff.net.0.proj.bias',
                    'layer2.weight': 'ff.net.2.weight',
                    'layer2.bias': 'ff.net.2.bias',
                }
                if sub in mapping:
                    out[f'transformer_blocks.{blk}.{mapping[sub]}'] = tensor
                continue

    # The diffusers CosmosTransformer3DModel always creates learnable_pos_embed
    # parameters (pos_emb_h, pos_emb_t, pos_emb_w) even though the ComfyUI
    # checkpoint doesn't include them. Initialize them to zeros so loading
    # doesn't fail with "Missing keys". Since the model uses RoPE for attention,
    # these per-block absolute positional embeddings have no effect when zeroed.
    if 'learnable_pos_embed.pos_emb_h' not in out:
        max_t = 128  # matches max_size[0]
        max_hw = 120  # matches max_size[1] // patch_size[1]
        model_ch = 2048  # attention_head_dim * num_attention_heads
        out['learnable_pos_embed.pos_emb_t'] = torch.zeros(max_t, model_ch)
        out['learnable_pos_embed.pos_emb_h'] = torch.zeros(max_hw, model_ch)
        out['learnable_pos_embed.pos_emb_w'] = torch.zeros(max_hw, model_ch)

    return out


# ---------------------------------------------------------------------------
# model_factory — main entry point
# ---------------------------------------------------------------------------
class model_factory:
    def __init__(
        self,
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

        with init_empty_weights():
            transformer = CosmosTransformer3DModel(**config)

        # Attach LLM adapter so its weights load via load_state_dict
        _attach_llm_adapter(transformer, config.get("text_embed_dim", 1024))

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

        # --- Text encoder (Qwen3 0.6B) ---
        import safetensors.torch as _st

        # Locate the Qwen3-0.6B model directory (has config.json + tokenizer)
        te_model_dir = None
        for _cp in fl._checkpoints_paths:
            _cand = os.path.join(_cp, "Qwen3-0.6B")
            if os.path.isdir(_cand) and os.path.isfile(os.path.join(_cand, "config.json")):
                te_model_dir = _cand
                break
        if not te_model_dir:
            raise FileNotFoundError("Qwen3-0.6B model directory not found in checkpoint paths")

        te_config = AutoConfig.from_pretrained(te_model_dir, trust_remote_code=True, local_files_only=True)
        if text_encoder_filename is None:
            raise ValueError("text_encoder_filename not provided")
        te_sd = _st.load_file(text_encoder_filename)
        te_sd["lm_head.weight"] = torch.zeros(te_config.vocab_size, te_config.hidden_size)
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(te_config)
        text_encoder.load_state_dict(te_sd, strict=False, assign=True)
        text_encoder.to(dtype).to("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder.eval()

        _orig_forward = text_encoder.forward
        def _te_forward(*a, **kw):
            kw.setdefault("output_hidden_states", True)
            out = _orig_forward(*a, **kw)
            if hasattr(out, "hidden_states") and out.hidden_states:
                out.last_hidden_state = out.hidden_states[-1]
            return out
        text_encoder.forward = _te_forward

        # --- Tokenizers ---
        # Qwen3 tokenizer for text encoding (from local model dir)
        tokenizer = AutoTokenizer.from_pretrained(te_model_dir, trust_remote_code=True, local_files_only=True)
        # T5 tokenizer for LLM adapter input (vocab_size=32128)
        t5_tok_dir = None
        for _cp in fl._checkpoints_paths:
            _cand = os.path.join(_cp, "t5-base-tokenizer")
            if os.path.isdir(_cand) and os.path.isfile(os.path.join(_cand, "tokenizer.json")):
                t5_tok_dir = _cand
                break
        if not t5_tok_dir:
            raise FileNotFoundError("t5-base-tokenizer directory not found in checkpoint paths")
        t5_tokenizer = AutoTokenizer.from_pretrained(t5_tok_dir, local_files_only=True, legacy=False)

        # --- VAE (Qwen-Image VAE via AutoencoderKLQwenImage) ---
        # Use the same VAE class as the Qwen Image models — the checkpoint
        # file qwen_image_vae.safetensors from circlestone-labs/Anima is in
        # ComfyUI format and does NOT match AutoencoderKLCosmos (0/310 keys).
        # Instead we download the diffusers-format VAE from DeepBeepMeep/Qwen_image.
        from models.qwen.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
        vae_filename = fl.locate_file("qwen_vae.safetensors", error_if_none=False)
        vae_config = fl.locate_file("qwen_vae_config.json", error_if_none=False)
        if vae_filename and vae_config:
            vae = offload.fast_load_transformers_model(
                vae_filename,
                writable_tensors=False,
                modelClass=AutoencoderKLQwenImage,
                defaultConfigPath=vae_config,
            )
        else:
            # Fallback: try AutoencoderKLCosmos (won't produce correct images
            # but at least won't crash during loading).
            logger.warning("qwen_vae.safetensors not found — falling back to AutoencoderKLCosmos")
            from diffusers import AutoencoderKLCosmos
            alt_vae = fl.locate_file("qwen_image_vae.safetensors", error_if_none=False)
            if alt_vae:
                vae_sd = _st.load_file(alt_vae)
                vae = AutoencoderKLCosmos()
                vae.load_state_dict(vae_sd, strict=False, assign=True)
            else:
                raise FileNotFoundError("No VAE file found")
        vae = vae.to(VAE_dtype)

        # --- Scheduler ---
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=False,
        )

        # --- Store references ---
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.vae = vae
        self.scheduler = scheduler
        self.pipeline = self
        self._interrupt_flag = False

    # ------------------------------------------------------------------
    # Text encoding with LLM adapter
    # ------------------------------------------------------------------
    def _encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Encode text through Qwen3 → LLM adapter → padded embeddings."""
        # 1. Qwen3 encoding
        qwen_ids = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=256, truncation=True,
        ).input_ids
        # Move token IDs to same device as text encoder (mmgp may place it on GPU)
        te_device = next(self.text_encoder.parameters()).device
        qwen_ids = qwen_ids.to(te_device)
        with torch.no_grad():
            te_out = self.text_encoder(input_ids=qwen_ids, output_hidden_states=True)
        qwen_hidden = te_out.hidden_states[-1]  # [1, 256, 1024]

        # 2. T5 tokenization (no special tokens, matching ComfyUI)
        t5_ids = self.t5_tokenizer.encode(text, add_special_tokens=False)
        # Both T5 IDs and Qwen3 hidden states must be on the adapter's device
        # (may be CPU if mmgp hasn't offloaded the adapter to GPU yet)
        adapter_device = next(self.transformer.llm_adapter.parameters()).device
        qwen_hidden = qwen_hidden.to(adapter_device)
        t5_ids_tensor = torch.tensor(t5_ids, dtype=torch.long, device=adapter_device).unsqueeze(0)

        # 3. LLM adapter: cross-attend T5 tokens to Qwen3 hidden states
        with torch.no_grad():
            adapter_out = self.transformer.llm_adapter(
                qwen_hidden, t5_ids_tensor
            )  # [1, t5_len, 1024]
        adapter_out = adapter_out.to(device)  # Move to target device for transformer

        # 4. Pad to 512 tokens (matching ComfyUI's preprocess_text_embeds)
        if adapter_out.shape[1] < 512:
            pad_len = 512 - adapter_out.shape[1]
            adapter_out = F.pad(adapter_out, (0, 0, 0, pad_len))

        return adapter_out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        seed=None,
        input_prompt="",
        n_prompt=None,
        sampling_steps=30,
        width=512,
        height=512,
        guide_scale=4.0,
        batch_size=1,
        callback=None,
        max_sequence_length=512,
        VAE_tile_size=None,
        loras_slists=None,
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.transformer.dtype

        # NOTE: Do NOT call self.transformer.to(device) — mmgp's offload system
        # hooks the model and creates circular module references that cause
        # infinite recursion in nn.Module._apply(). mmgp handles device
        # placement automatically during forward passes.

        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                self.vae.use_tiling = VAE_tile_size > 0
                self.vae.tile_latent_min_height = max(VAE_tile_size, 0)
                self.vae.tile_latent_min_width = max(VAE_tile_size, 0)

        generator = torch.Generator()
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        torch.cuda.empty_cache()
        gc.collect()

        # Encode text via Qwen3 → LLM adapter
        neg = n_prompt or "worst quality, low quality, score_1, score_2, score_3, artist name"
        neg_embeds = self._encode_text(neg, device)
        pos_embeds = self._encode_text(input_prompt, device)

        # Create latents — x_0 is noise for flow matching (CONST type)
        # Anima flow matching (matching ComfyUI's ModelSamplingDiscreteFlow):
        #   sigma_schedule: time_snr_shift(3.0, t) for t in linspace(1/steps, 1, steps)
        #   forward:        x_t = sigma * noise + (1 - sigma) * x_0
        #   timestep to model: sigma (multiplier=1.0, so timestep == sigma)
        #   model predicts:  velocity v
        #   denoised:        x_0 = x_t - sigma * v
        #   euler step:      x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v
        vae_scale = 8
        latent_h, latent_w = height // vae_scale, width // vae_scale
        latents = torch.randn(
            1, 16, latent_h, latent_w, generator=generator,
            device="cpu", dtype=dtype,
        ).to(device)

        # Compute sigma schedule: time_snr_shift(3.0, t) for t in [1, 0]
        def _time_snr_shift(alpha, t):
            if alpha == 1.0:
                return t
            return alpha * t / (1.0 + (alpha - 1.0) * t)

        # "simple" scheduler: evenly spaced from sigma_max to sigma_min
        t_lin = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)
        sigmas = torch.tensor(
            [_time_snr_shift(3.0, t.item()) for t in t_lin],
            device=device, dtype=dtype,
        )
        # sigma[0] = sigma_max ≈ 1.0, sigma[-1] = 0.0

        padding_mask = torch.ones(
            1, 1, latent_h, latent_w, device=device, dtype=dtype,
        )

        # Denoising loop: iterate from sigma_max → 0
        for i in range(sampling_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Timestep = sigma (multiplier=1.0 in ComfyUI's ModelSamplingDiscreteFlow)
            timestep = sigma_cur.expand(2)

            with torch.no_grad():
                latent_in = torch.cat([latents] * 2).unsqueeze(2)  # [2, C, 1, H, W]
                embeds = torch.cat([neg_embeds, pos_embeds])
                velocity = self.transformer(
                    hidden_states=latent_in,
                    encoder_hidden_states=embeds,
                    timestep=timestep,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

            np_u, np_t = velocity.chunk(2)
            del velocity, latent_in, embeds
            velocity_cfg = np_u + guide_scale * (np_t - np_u)
            del np_u, np_t

            # Flow matching Euler step: x_{next} = x_cur + (sigma_next - sigma_cur) * v
            # Since sigma_next < sigma_cur, this moves toward x_0
            latents = latents + (sigma_next - sigma_cur) * velocity_cfg.squeeze(2)
            del velocity_cfg
            torch.cuda.empty_cache()

        # Let mmgp handle offloading the transformer before VAE decode —
        # calling self.transformer.to("cpu") triggers infinite recursion in
        # nn.Module._apply() because mmgp hooks create circular references.
        torch.cuda.empty_cache()
        gc.collect()

        # Decode latents → image
        # The Qwen-Image VAE expects denormalized latents:
        #   latents_denorm = latents * std + mean
        # The diffusion process produces normalized latents, so we must undo
        # the normalization that was applied during encoding.
        with torch.no_grad():
            # mmgp moves the VAE to GPU automatically during decode — do not
            # call self.vae.to(device) (same recursion issue as transformer).
            latents_5d = latents.float().unsqueeze(2)  # [B, C, 1, H, W]
            if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'latents_mean'):
                vae_z_dim = latents_5d.shape[1]
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, vae_z_dim, 1, 1, 1)
                    .to(latents_5d.device, latents_5d.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, vae_z_dim, 1, 1, 1)
                    .to(latents_5d.device, latents_5d.dtype)
                )
                latents_5d = latents_5d * latents_std + latents_mean

            # Decode using the standard decode path (returns float tensor)
            # decode_to_cpu_uint8 can return None in some edge cases
            decoded = self.vae.decode(latents_5d, return_dict=False)[0]
            image = decoded.squeeze(2)  # [B, C, H, W] — remove temporal dim
            image = image.clamp(-1, 1)  # VAE output is typically [-1, 1]
            return image[0]  # [C, H, W] float

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return self._interrupt_flag

    @_interrupt.setter
    def _interrupt(self, value):
        self._interrupt_flag = value
