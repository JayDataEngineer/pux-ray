"""MOSS-SoundEffect raw nn.Module loading + mmgp setup.

Decomposes the 8B model into:
- language_model: Qwen3Model (36 layers, 4096 hidden)
- emb_ext: 16 nn.Embedding layers (VQ audio code embeddings)
- lm_heads: 17 nn.Linear heads (1 text + 16 audio VQ)

Plus CPU-side audio tokenizer:
- audio_encoder: PatchedPretransform + Transformer stack (encoder)
- quantizer: MossAudioTokenizerResidualLFQ (32 quantizers, 1024 codes)
- audio_decoder: Transformer + PatchedPretransform stack (decoder)

The processor (MossTTSDelayProcessor) handles tokenization and delay patterns.
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

MODELS_ROOT = os.environ.get("TECH_NOIR_MODELS_ROOT", "/home/user/Documents/models")
MODEL_PATH = os.environ.get("MOSS_SFX_MODEL_PATH", os.path.join(MODELS_ROOT, "audio/moss-soundeffect"))


@dataclass
class MossModules:
    """All raw nn.Modules for MOSS-SoundEffect inference."""

    # Full model reference — needed for generate() (delay pattern coupling)
    model: Any

    # Main model components (GPU) — for mmgp pipe dict
    language_model: Any
    emb_ext: Any          # nn.ModuleList of 16 nn.Embedding
    lm_heads: Any         # nn.ModuleList of 17 nn.Linear

    # Audio tokenizer (CPU)
    audio_tokenizer: Any

    # Processor (handles tokenization + delay patterns)
    processor: Any
    config: Any
    device: torch.device = torch.device("cuda")

    # mmgp
    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MossModules:
        from services.compat import apply
        apply()
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)

        model_path_str = str(model_path)
        if not os.path.isdir(model_path_str):
            raise FileNotFoundError(f"MOSS model not found at {model_path_str}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        actual_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Pre-patch model class before from_pretrained
        _patch_moss_model_class(model_path_str)

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(
            model_path_str, trust_remote_code=True, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_str, trust_remote_code=True, local_files_only=True,
        )

        # Load audio tokenizer on CPU
        codec_path = os.environ.get(
            "MOSS_AUDIO_TOKENIZER_PATH",
            os.path.join(MODELS_ROOT, "audio/moss-audio-tokenizer"),
        )
        logger.info("Loading MOSS audio tokenizer from %s (CPU)", codec_path)
        audio_tokenizer = AutoModel.from_pretrained(
            codec_path, trust_remote_code=True, local_files_only=True,
            device_map="cpu", torch_dtype=torch.float32,
        )

        # Build processor manually
        processor = _build_processor(model_path_str, config, tokenizer, audio_tokenizer)

        # Load main model
        config.tie_word_embeddings = False
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Loading MOSS model (%s) from %s", actual_dtype, model_path_str)
        model = AutoModel.from_pretrained(
            model_path_str,
            config=config,
            trust_remote_code=True,
            torch_dtype=actual_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": 0} if device == "cuda" else None,
        )
        model.eval()

        # Extract sub-modules from the model
        language_model = getattr(model, "language_model", model)
        emb_ext = getattr(model, "emb_ext", None)
        lm_heads = getattr(model, "lm_heads", None)

        # If we can't extract sub-modules, use the whole model
        if emb_ext is None or lm_heads is None:
            logger.warning("Could not extract emb_ext/lm_heads — using full model in pipe")
            pipe = {"model": model}
        else:
            # Put language model + heads in pipe for mmgp, keep embeddings separately
            pipe = {
                "language_model": language_model,
                "lm_heads": lm_heads,
            }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if device == "cuda" else 0
        logger.info("MOSS loaded on %s (VRAM: %.0fMB)", device, vram)

        return cls(
            model=model,
            language_model=language_model,
            emb_ext=emb_ext,
            lm_heads=lm_heads,
            audio_tokenizer=audio_tokenizer,
            processor=processor,
            config=config,
            device=torch.device(device),
            pipe=pipe,
            co_tenants={},
        )


def _patch_moss_model_class(model_path: str):
    """Patch get_input_embeddings before from_pretrained instantiates the class."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    auto_map = getattr(config, "auto_map", {})
    class_ref = auto_map.get("AutoModel")
    if not class_ref:
        return

    cls = get_class_from_dynamic_module(class_ref, model_path, trust_remote_code=True)
    orig = cls.get_input_embeddings
    sig = inspect.signature(orig)
    required = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
    if len(required) > 1:
        def _compat(self_inner, input_ids=None):
            if input_ids is None:
                lm = getattr(self_inner, "language_model", None)
                if lm is not None:
                    emb = lm.get_input_embeddings()
                    if emb is not None:
                        return emb
                return self_inner
            return orig(self_inner, input_ids)
        cls.get_input_embeddings = _compat


def _build_processor(model_path, config, tokenizer, audio_tokenizer):
    """Build MossTTSDelayProcessor manually."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    proc_cls = get_class_from_dynamic_module(
        "processing_moss_tts.MossTTSDelayProcessor",
        model_path, trust_remote_code=True,
    )
    processor = proc_cls.__new__(proc_cls)
    processor.tokenizer = tokenizer
    processor.audio_tokenizer = audio_tokenizer

    if config is None:
        from importlib import import_module
        cfg_mod = import_module(
            "transformers_modules.moss_hyphen_soundeffect.configuration_moss_tts"
        )
        config = cfg_mod.MossTTSDelayConfig()
    processor.model_config = config

    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id

    processor.imstart_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    processor.imend_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    processor.newline_token_id = 198

    def _id_to_token(token_id):
        tok = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(tok, list):
            return tok[0] if len(tok) > 0 else ""
        return tok

    processor.audio_user_slot_token = _id_to_token(config.audio_user_slot_token_id)
    processor.audio_assistant_gen_slot_token = _id_to_token(config.audio_assistant_gen_slot_token_id)
    processor.audio_assistant_delay_slot_token = _id_to_token(config.audio_assistant_delay_slot_token_id)
    processor.audio_start_token = _id_to_token(config.audio_start_token_id)
    processor.audio_end_token = _id_to_token(config.audio_end_token_id)

    return processor
