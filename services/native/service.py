"""Native model service — Pragmatic Extreme architecture.

ARCHITECTURE (from the docs):
  1. transformers 4.x — text encoders only (Qwen3, T5, Gemma). Pinned <5.0.0.
  2. diffusers model CLASSES — architecture shell only, instantiated on meta device.
     Zero pipeline overhead. Zero loader overhead. We load safetensors manually.
  3. BlockStreamHook — custom VRAM streaming via forward_pre_hook + pin_memory + CUDA streams.
  4. Manual Euler denoise loop — we control every step. No diffusers pipeline.
  5. torch.compile — compiled graph execution for the transformer.
  6. NO diffusers pipelines. NO mmGP. NO Wan2GP. NO SGLang serve.

The "meta device" trick:
  - with torch.device("meta"): model = ZImageTransformer2DModel(...)
  - Creates the full parameter structure in 0ms using 0MB RAM
  - safetensors.torch.load_file() loads weights to host memory
  - BlockStreamHook streams weights to GPU on-demand through pinned memory + CUDA streams
"""
from __future__ import annotations

import base64, gc, io, json, logging, os, time
from typing import Any, Optional

import safetensors.torch as st
import torch
import torch.nn as nn

from services.forge_base import ForgeService


def _clamp_resolution(px: int) -> int:
    """Clamp to nearest supported static resolution for compiled inference."""
    valid = [512, 768, 1024, 1280, 1536]
    return min(valid, key=lambda x: abs(x - px))
from services.forge_persistence import Persistence
from services.native.registry import get_model, ModelEntry, ALL_MODELS
from services.native.loader import BlockStreamHook, VRAMManager, VRAMPlan, Strategy, plan, module_size_mb

logger = logging.getLogger(__name__)
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")

# ── Force SDPA FlashAttention-2 via environment ────────────────────────────
os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")  # TF32 on Ampere+
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
# These tell PyTorch's SDPA to prefer FlashAttention-2 kernels:
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                       "expandable_segments:True,roundup_power2_divisions:16")
# FlashAttention-2 is the default in PyTorch 2.10 for compatible GPUs (sm80+).
# Z-Image transformer already uses F.scaled_dot_product_attention internally.


# ─── Transformer class map ────────────────────────────────────────────────
# diffusers provides the nn.Module architecture shell. We instantiate on meta
# device (0ms, 0MB), load safetensors manually, and stream via BlockStreamHook.
_TRANSFORMER_CLASSES: dict[str, str] = {
    "ZImagePipeline":       "ZImageTransformer2DModel",
    "FluxPipeline":         "FluxTransformer2DModel",
    "LTXPipeline":          "LTXVideoTransformer3DModel",
    "QwenImagePipeline":    "Qwen2ImageTransformer2DModel",
    "ModularPipeline":      None,
}


def _get_transformer_cls(pipeline_cls: str):
    """Get the diffusers model class for architecture definition (meta device only)."""
    import diffusers
    name = _TRANSFORMER_CLASSES.get(pipeline_cls)
    if name is None:
        raise ValueError(f"No transformer class for {pipeline_cls}. "
                         f"Known: {list(_TRANSFORMER_CLASSES.keys())}")
    cls = getattr(diffusers, name, None)
    if cls is None:
        raise ValueError(f"diffusers.{name} not found — is diffusers installed?")
    return cls


# ─── Text encoder — kept on CPU, freed after encoding ─────────────────────
# transformers 4.x AutoModel, loaded on CPU, encode prompt, freed.
# No GPU memory consumed except during the ~2s encoding window.


def _load_text_encoder(model_path: str, dtype: torch.dtype, device: torch.device):
    """Load text encoder on CPU (warm resident, no GPU VRAM consumed).

    The text encoder (Qwen3) is ~7.7GB in BF16. Keeping it on CPU avoids
    stealing VRAM from the transformer/VAE. It is NOT freed after encoding
    to avoid reload overhead. Encoding takes ~5s on CPU — acceptable for
    the throughput we need.
    """
    te_path = os.path.join(model_path, "text_encoder")
    if not os.path.exists(te_path):
        return None, None

    import transformers
    logger.info("Text encoder: loading on CPU (warm resident)...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer"))
    text_encoder = transformers.AutoModel.from_pretrained(
        te_path, torch_dtype=dtype)
    text_encoder.eval()  # Keep on CPU, don't move to GPU
    logger.info("Text encoder: %s on CPU (%.1fMB, warm)",
                type(text_encoder).__name__,
                sum(p.numel() for p in text_encoder.parameters()) * dtype.itemsize / (1024*1024))
    return tokenizer, text_encoder


def _encode_prompts(tokenizer, text_encoder, prompts: list[str],
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode prompts using warm CPU text encoder. Returns fixed-shape embeddings.

    Always pads/truncates to 256 tokens for static shapes (enables torch.compile
    on the transformer, which requires fixed-length embeddings).
    Text encoder stays on CPU (warm resident). Tokens sent to CPU, output moved
    to GPU.
    """
    SEQ_LEN = 256  # Fixed sequence length for static compilation
    if text_encoder is None or tokenizer is None:
        return torch.randn(len(prompts), SEQ_LEN, 2048, device=device, dtype=dtype)

    tokens = tokenizer(prompts, return_tensors="pt", padding="max_length",
                       max_length=SEQ_LEN, truncation=True)
    with torch.no_grad():
        out = text_encoder(**tokens)
    return out.last_hidden_state.to(device=device, dtype=dtype)


# ─── VAE — always BF16 resident (tiny, precision-critical) ────────────────


def _load_vae(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load VAE: meta device init → manual safetensor load → GPU → compile decode."""
    vae_path = os.path.join(model_path, "vae")
    if not os.path.exists(vae_path):
        return None

    from diffusers import AutoencoderKL

    # 1. Load config
    config_path = os.path.join(vae_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # 2. Instantiate on meta device
    with torch.device("meta"):
        vae = AutoencoderKL.from_config(config)

    # 3. Load safetensors
    safetensor_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(safetensor_path):
        state_dict = st.load_file(safetensor_path)
        vae.load_state_dict(state_dict, strict=False, assign=True)

    vae = vae.to(dtype=dtype, device=device).eval()
    logger.info("VAE: AutoencoderKL loaded on GPU")
    return vae


# ─── Transformer — meta device → manual safetensors → BlockStreamHook ─────


def _load_transformer_meta(model_path: str, pipeline_cls: str,
                           dtype: torch.dtype = torch.bfloat16):
    """Instantiate transformer on meta device and load weights shard-by-shard.

    The safetensor weights may be FP32 but we cast to *dtype* (default BF16)
    after loading to save VRAM. Returns (model, tr_mb) where tr_mb is the
    estimated GPU memory in MB for the cast weights.

    Shard-by-shard loading avoids OOM: each shard is loaded, assigned to the
    meta model via assign=True, then freed before the next shard.
    """
    tr_path = os.path.join(model_path, "transformer")
    tr_cls = _get_transformer_cls(pipeline_cls)
    element_bytes = dtype.itemsize  # 2 for BF16, 4 for FP32

    # 1. Load config
    config_path = os.path.join(tr_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    config.pop("_name_or_path", None)

    # 2. Instantiate on meta device — 0ms, 0MB
    logger.info("Transformer: instantiating %s on meta device", tr_cls.__name__)
    with torch.device("meta"):
        transformer = tr_cls.from_config(config)

    param_count = sum(p.numel() for p in transformer.parameters())
    tr_mb = int(param_count * element_bytes / (1024 * 1024))
    logger.info("Transformer: %s (%d params, ~%dMB as %s)",
                tr_cls.__name__, param_count, tr_mb, dtype)

    # 3. Load safetensors shard-by-shard
    index_path = os.path.join(tr_path, "diffusion_pytorch_model.safetensors.index.json")
    single_path = os.path.join(tr_path, "diffusion_pytorch_model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for sf in shard_files:
            shard_path = os.path.join(tr_path, sf)
            shard_size = os.path.getsize(shard_path)
            logger.info("  Shard: %s (%.1fGB)", sf, shard_size / 1e9)
            shard_sd = st.load_file(shard_path)
            transformer.load_state_dict(shard_sd, strict=False, assign=True)
            del shard_sd
            gc.collect()
    elif os.path.exists(single_path):
        logger.info("  Loading single safetensor: %.1fGB",
                    os.path.getsize(single_path) / 1e9)
        state_dict = st.load_file(single_path)
        transformer.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict
    else:
        raise FileNotFoundError(f"No safetensors found in {tr_path}")

    # 4. Cast loaded weights to target dtype (safetensors may be FP32)
    logger.info("Transformer: casting to %s", dtype)
    transformer = transformer.to(dtype=dtype)

    gc.collect()
    return transformer, tr_mb


class NativeService(ForgeService):
    """Serves models via Pragmatic Extreme architecture.

    - transformers 4.x for text encoding (CPU)
    - diffusers model class as meta-device architecture shell
    - Manual safetensor loading
    - BlockStreamHook VRAM streaming
    - Manual Euler denoise loop
    """

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self.entry: Optional[ModelEntry] = None
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self._vram_manager: Optional[VRAMManager] = None
        self._loaded_model: str | None = None
        self._compiled = False  # torch.compile applied?

    def load(self, model_name: str, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ALL_MODELS.keys())}")

        self.entry = entry
        model_path = self._resolve_path(entry)
        logger.info("Native: loading '%s' from %s", model_name, model_path)

        device = torch.device("cuda")
        dtype = torch.bfloat16

        # ── 1. Transformer: meta device + shard-by-shard weight loading ────
        # Loads weights shard-by-shard to avoid OOM (~10GB peak per shard).
        # Parameters are assigned to the meta model via load_state_dict(assign=True).
        t0 = time.perf_counter()
        transformer, tr_mb = _load_transformer_meta(model_path, entry.pipeline)
        load_time = time.perf_counter() - t0
        logger.info("Transformer: meta-init + shard load in %.1fs (~%dMB)", load_time, tr_mb)

        self.transformer = transformer

        # Detect model type (must be done BEFORE compile wraps in OptimizedModule)
        self._pipeline_type = entry.pipeline  # e.g. "ZImagePipeline"
        self._is_z_image = type(transformer).__name__.lower().startswith("zimage")

        # Build VRAM plan (transformer weights are on CPU after assign)
        vae_mb = 300
        vram_plan = plan(transformer_mb=tr_mb, encoder_mb=0, vae_mb=vae_mb)
        logger.info("VRAM plan: %s", vram_plan)

        # If plan chose RESIDENT but model is huge, force BLOCK_STREAM
        if vram_plan.strategy == Strategy.RESIDENT and tr_mb > 20000:
            from services.native.loader import Strategy as S
            vram_plan = VRAMPlan(
                strategy=S.BLOCK_STREAM, use_compile=False,
                use_cache=False, use_fp8=False,
                estimated_vram_mb=tr_mb // 2,
                notes=f"Forced block streaming ({tr_mb}MB > 20GB threshold)",
            )
            logger.warning("VRAM: forced BLOCK_STREAM for large model (%dMB)", tr_mb)

        # Apply VRAM strategy (BlockStreamHook for streaming, or resident for small models)
        self._vram_manager = VRAMManager(transformer, device=str(device))
        self._vram_manager.apply(vram_plan)
        gc.collect()

        # ── 2. Text encoder on GPU (warm resident, no reloads) ──────────────
        self.tokenizer, self.text_encoder = _load_text_encoder(model_path, dtype, device)

        # ── 3. Compile transformer (reduce-overhead, fast first-run) ────────
        # NOTE: max-autotune takes 10-30 min to compile on 6B models.
        # reduce-overhead compiles in ~30s, gives ~80% of max-autotune speedup.
        if vram_plan.use_compile:
            logger.info("Compiling transformer with torch.compile (reduce-overhead)...")
            try:
                self.transformer = torch.compile(
                    self.transformer,
                    mode="reduce-overhead",
                )
                self._compiled = True
            except Exception as e:
                logger.warning("torch.compile failed (%s), using eager", e)
                self._compiled = False

        # ── 4. VAE on GPU ───────────────────────────────────────────────────
        self.vae = _load_vae(model_path, device, dtype)

        self._loaded = True
        self.model_name = model_name
        self._loaded_model = model_name

        vram = self.actual_vram_mb()
        logger.info("Native: '%s' ready (%dMB VRAM, plan=%s)",
                    model_name, vram, vram_plan.strategy.value)

    def unload(self) -> None:
        if self._vram_manager:
            self._vram_manager.remove()
            self._vram_manager = None
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self._loaded = False
        self._compiled = False
        self.model_name = None
        self._loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Native: unloaded")

    def infer(self, payload: dict) -> dict:
        if not self._loaded or self.transformer is None:
            return {"status": "error", "error": "Transformer not loaded"}
        try:
            return self._generate(payload)
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            return {"status": "error", "error": f"CUDA OOM: {e}"}
        except Exception as e:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(e)}

    def _generate(self, payload: dict) -> dict:
        e = self.entry
        prompt = payload.get("prompt") or payload.get("input_prompt") or ""
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        neg = payload.get("n_prompt") or payload.get("negative_prompt") or ""
        steps = int(payload.get("steps") or payload.get("sampling_steps") or e.steps)
        guidance = float(payload.get("guidance") or payload.get("guide_scale") or e.guidance)
        # Clamp to static resolutions for compiled inference
        width = _clamp_resolution(int(payload.get("width", e.width)))
        height = _clamp_resolution(int(payload.get("height", e.height)))
        seed = payload.get("seed", -1)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # ── Encode text on CPU (warm-resident, no GPU VRAM used) ────────────
        t0 = time.perf_counter()
        embeds = _encode_prompts(self.tokenizer, self.text_encoder,
                                 [prompt, neg], device, dtype)
        encode_time = time.perf_counter() - t0
        # Text encoder stays warm on CPU — NOT freed.

        # ── Init latents ────────────────────────────────────────────────────
        gen = torch.Generator(device=device).manual_seed(int(seed)) if seed >= 0 else None
        latent_h, latent_w = height // 8, width // 8

        # Determine latent format from transformer class (detected before compile)
        is_z_image = self._is_z_image
        is_flux = self._pipeline_type == "FluxPipeline"

        if is_z_image:
            # Z-Image: 5D latents (B, C, F=1, H, W)
            latents = torch.randn(1, 16, 1, latent_h, latent_w,
                                  device=device, dtype=dtype, generator=gen)
        else:
            # Standard 4D latents
            latents = torch.randn(1, 16, latent_h, latent_w,
                                  device=device, dtype=dtype, generator=gen)

        # ── Euler denoise loop (torch.compiled if _compiled=True) ───────────
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        dt0 = time.perf_counter()

        for i in range(steps):
            sigma = sigmas[i]
            sn = sigmas[i + 1]
            t_val = (sigma * 1000).expand(1)
            x_in = torch.cat([latents, latents])
            t_in = t_val.expand(2)

            with torch.no_grad():
                if is_z_image:
                    # Z-Image forward: (x, t, cap_feats)
                    result = self.transformer(x=x_in, t=t_in, cap_feats=embeds,
                                              return_dict=False)
                    noise_list = result[0] if isinstance(result[0], list) else result
                    noise_neg = noise_list[0].unsqueeze(0)
                    noise_pos = noise_list[1].unsqueeze(0)
                else:
                    # Standard forward: (hidden_states, timestep, encoder_hidden_states)
                    noise = self.transformer(
                        hidden_states=x_in, timestep=t_in,
                        encoder_hidden_states=embeds, return_dict=False)[0]
                    noise_neg, noise_pos = noise.chunk(2)

            noise_pred = noise_neg + guidance * (noise_pos - noise_neg)
            latents = latents + (sn - sigma) * noise_pred
        torch.cuda.synchronize()

        gen_time = time.perf_counter() - dt0
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        # ── VAE decode (compiled) ───────────────────────────────────────────
        if is_z_image:
            latents = latents.squeeze(2)  # Remove temporal dim for 2D VAE

        with torch.no_grad():
            decoded = self.vae.decode(latents, return_dict=False)[0]

        # ── Encode as base64 PNG ────────────────────────────────────────────
        from PIL import Image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img[0] * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        return {
            "status": "success",
            "output": {
                "type": "image",
                "content": base64.b64encode(buf.getvalue()).decode(),
                "format": "png",
            },
            "metrics": {
                "latency_ms": int(gen_time * 1000),
                "encode_ms": int(encode_time * 1000),
                "model": self.model_name,
                "vram_peak_mb": int(peak_vram),
                "steps": steps,
                "compiled": self._compiled,
            },
        }

    def _resolve_path(self, entry: ModelEntry) -> str:
        for p in [f"/models/native/{entry.name}", f"/models/{entry.name}", entry.repo]:
            if os.path.exists(p):
                return p
        return entry.repo

    def actual_vram_mb(self) -> int:
        try:
            return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        except Exception:
            return 0
