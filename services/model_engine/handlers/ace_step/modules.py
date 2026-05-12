"""ACE-Step v1.5 raw nn.Module loading + mmgp setup.

Loads components individually (not as a pipeline), extracts codebook/
projection weights for manual audio-code → latent conversion, and
prepares the pipe dict for mmgp VRAM management.

Reference: Wan2GP's ace_step_handler.py + models/TTS/ace_step15/
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# ── Transformer config variants ────────────────────────────────────────────

V15_VARIANTS = {
    "base": "ace_step_v1_5_transformer_config_base.json",
    "sft": "ace_step_v1_5_transformer_config_sft.json",
    "turbo": "ace_step_v1_5_transformer_config_turbo.json",
    "turbo_shift1": "ace_step_v1_5_transformer_config_turbo_shift1.json",
    "turbo_shift3": "ace_step_v1_5_transformer_config_turbo_shift3.json",
    "turbo_continuous": "ace_step_v1_5_transformer_config_turbo_continuous.json",
    "xl_turbo": "ace_step_v1_5_xl_transformer_config_turbo.json",
}


# ── Module container ───────────────────────────────────────────────────────

@dataclass
class AceStepModules:
    """All raw nn.Modules needed for ACE-Step inference.

    Every module is loaded independently and managed by mmgp.
    The orchestrator calls .forward() directly — no pipeline wrapper.
    """

    transformer: Any
    text_encoder_2: Any
    codec: Any
    lm_model: Any
    pre_text_tokenizer: Any
    lm_tokenizer: Any
    dtype: torch.dtype
    device: torch.device

    # Codebook extraction (bypasses mmgp-incompatible quantizer object)
    codebook: Optional[torch.Tensor] = None
    proj_weight: Optional[torch.Tensor] = None
    proj_bias: Optional[torch.Tensor] = None
    detokenizer: Optional[Any] = None

    # Audio code vocabulary (from LM tokenizer)
    audio_code_token_ids: Optional[torch.Tensor] = None
    audio_code_token_map: Optional[dict] = None
    audio_code_mask: Optional[torch.Tensor] = None

    # Silence latent (pre-computed)
    silence_latent: Optional[torch.Tensor] = None

    # MMGP pipe dict + co-tenants
    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_type: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_lm: bool = True,
    ) -> AceStepModules:
        variant_name = _resolve_variant(model_type)
        config_filename = V15_VARIANTS.get(variant_name, V15_VARIANTS["turbo"])

        logger.info("Loading ACE-Step v1.5 variant=%s config=%s", variant_name, config_filename)

        # Resolve component directories
        transformer_dir = _find_dir(model_path, ["acestep-v15-turbo", "acestep-v1.5-turbo", variant_name])
        vae_dir = _find_dir(model_path, ["vae"])
        te2_dir = _find_dir(model_path, ["Qwen3-Embedding-0.6B", "qwen3-embedding"])
        lm_dir = _find_dir(model_path, ["acestep-5Hz-lm-1.7B", "acestep-lm"])

        # Import model classes
        from .models.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
        from .models.configuration_acestep_v15 import AceStepConfig
        from diffusers import AutoencoderOobleck
        from transformers import Qwen3Model, Qwen3ForCausalLM, AutoTokenizer

        # ── Load transformer ────────────────────────────────────────
        config_path = transformer_dir / config_filename
        if not config_path.exists():
            config_path = transformer_dir / "config.json"

        logger.info("Loading transformer from %s", transformer_dir)
        transformer = _fast_load(
            _find_weights(transformer_dir), AceStepConditionGenerationModel,
            config_path, dtype,
        )
        transformer.eval()

        # ── Load VAE ─────────────────────────────────────────────────
        vae_weights = _find_weights(vae_dir)
        logger.info("Loading VAE from %s", vae_weights.name)
        audio_vae = _fast_load(vae_weights, AutoencoderOobleck, vae_dir / "config.json", dtype)
        audio_vae.eval()

        # ── Load text encoder 2 (Qwen3 0.6B embedding) ──────────────
        logger.info("Loading text encoder 2 from %s", te2_dir)
        text_encoder_2 = _fast_load(
            _find_weights(te2_dir), Qwen3Model, te2_dir / "config.json", dtype,
        )
        text_encoder_2.eval()

        # ── Load LM (optional) ───────────────────────────────────────
        lm_model = None
        if enable_lm and lm_dir is not None and lm_dir.is_dir():
            logger.info("Loading LM from %s", lm_dir)
            lm_model = _fast_load(
                _find_weights(lm_dir), Qwen3ForCausalLM, lm_dir / "config.json", dtype,
            )
            lm_model.eval()
            # Move LM to device explicitly — mmgp's pinning has issues with
            # tied weights (embed_tokens <-> lm_head) and generate() hooks.
            lm_model = lm_model.to(device)

        # ── Silence latent ───────────────────────────────────────────
        silence_latent = _find_and_load_silence(model_path, transformer_dir)

        # ── Tokenizers ───────────────────────────────────────────────
        pre_text_tokenizer = AutoTokenizer.from_pretrained(
            str(te2_dir), padding_side="right",
        )
        lm_tokenizer = None
        if lm_model is not None:
            lm_tokenizer = AutoTokenizer.from_pretrained(
                str(lm_dir), padding_side="left",
            )

        # ── Build audio code vocabulary ──────────────────────────────
        audio_code_token_ids = audio_code_token_map = audio_code_mask = None
        if lm_tokenizer is not None:
            from .audio_codes import build_audio_code_vocab
            audio_code_token_ids, audio_code_token_map, audio_code_mask = (
                build_audio_code_vocab(lm_tokenizer)
            )

        # ── Extract codebook + projection weights ──────────────────
        codebook = proj_weight = proj_bias = detokenizer = None
        if hasattr(transformer, "tokenizer") and hasattr(transformer.tokenizer, "quantizer"):
            try:
                q = transformer.tokenizer.quantizer
                codebook = q.codebooks.detach().clone().cpu().float()
                proj_weight = q.project_out.weight.data.detach().clone().cpu().float()
                proj_bias = q.project_out.bias.data.detach().clone().cpu().float()
                d_copy = copy.deepcopy(transformer.detokenizer)
                detokenizer = d_copy.cpu().float()
            except Exception as e:
                logger.warning("Failed to extract codebook/projection: %s", e)

        # ── Build MMGP pipe dict ────────────────────────────────────
        # LM managed separately on GPU (not in mmgp pipe — tied weights + generate() hook issues)
        pipe = {"transformer": transformer, "text_encoder_2": text_encoder_2, "codec": audio_vae}
        co_tenants = {"transformer": ["codec"]}

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info(
            "ACE-Step v1.5 loaded: variant=%s VRAM=%.0fMB pipe_keys=%s lm=%s",
            variant_name, vram, list(pipe.keys()), "yes" if lm_model else "no",
        )

        return cls(
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            codec=audio_vae,
            lm_model=lm_model,
            pre_text_tokenizer=pre_text_tokenizer,
            lm_tokenizer=lm_tokenizer,
            dtype=dtype,
            device=torch.device(device),
            codebook=codebook,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            detokenizer=detokenizer,
            audio_code_token_ids=audio_code_token_ids,
            audio_code_token_map=audio_code_token_map,
            audio_code_mask=audio_code_mask,
            silence_latent=silence_latent,
            pipe=pipe,
            co_tenants=co_tenants,
        )


# ── Helpers (moved from __init__.py for module-level reuse) ────────────────

def _resolve_variant(model_type: str) -> str:
    if model_type == "ace_step_v1_5":
        return "base"
    if model_type == "ace_step_v1_5_xl_turbo":
        return "xl_turbo"
    parts = model_type.replace("ace_step_v1_5_", "")
    return parts if parts in V15_VARIANTS else "turbo"


def _find_dir(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_dir():
                for name in candidates:
                    if entry.name.lower() == name.lower():
                        return entry
    raise FileNotFoundError(f"Component not found in {root} (candidates: {candidates})")


def _find_weights(component_dir: Path) -> Path:
    for pattern in ["model.safetensors", "*.safetensors", "diffusion_pytorch_model.safetensors"]:
        matches = list(component_dir.glob(pattern))
        if matches:
            return matches[0]
    for pattern in ["*.bin", "*.pt", "*.pth"]:
        matches = list(component_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No weights file found in {component_dir}")


def _fast_load(
    weights_path: Path, model_class: type, config_path: Path, dtype: torch.dtype,
) -> torch.nn.Module:
    from safetensors.torch import load_file

    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        for key in [
            "_class_name", "_diffusers_version", "_name_or_path",
            "auto_map", "_transformers_version",
        ]:
            config_dict.pop(key, None)

        config_cls = getattr(model_class, "config_class", None)
        if config_cls is not None:
            try:
                config = config_cls(**config_dict)
                model = model_class(config)
            except (TypeError, Exception):
                model = None
        else:
            model = None

        if model is None and hasattr(model_class, "from_config"):
            try:
                model = model_class.from_config(config_dict)
            except Exception:
                model = None

        if model is None:
            try:
                model = model_class(**config_dict)
            except TypeError:
                model = model_class()
    else:
        model = model_class()

    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype)
    return model


def _find_and_load_silence(root: Path, transformer_dir: Path) -> Optional[torch.Tensor]:
    for base in [transformer_dir, root]:
        for name in ["silence_latent.pt", "silence_latents.pt"]:
            path = base / name
            if path.exists():
                return torch.load(str(path), map_location="cpu")
    return None
