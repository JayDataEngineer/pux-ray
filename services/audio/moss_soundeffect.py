"""MOSS-SoundEffect — Text-to-sound effect generation (Ray-native).

8B parameter model from the MOSS-TTS family. Generates environmental sounds,
urban scenes, creatures, human actions, and music-like clips from text prompts.
~16GB VRAM in bf16. Routes through master router for exclusive GPU access.
"""
from __future__ import annotations

import asyncio
import gc
import inspect
import io
import logging
import os
import time

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODELS_ROOT = os.environ.get("TECH_NOIR_MODELS_ROOT", "/home/user/Documents/models")
MODEL_PATH = os.environ.get("MOSS_SFX_MODEL_PATH", os.path.join(MODELS_ROOT, "audio/moss-soundeffect"))


def _patch_transformers():
    """Apply all transformers compat patches needed by MOSS."""
    from services.compat import apply
    apply()


def _get_moss_model_class():
    """Import and return the MOSS model class with get_input_embeddings patched.

    MOSS model's get_input_embeddings(self, input_ids) takes a required arg but
    transformers calls it without args during tie_weights. We pre-patch the class
    before from_pretrained instantiates it.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    auto_map = getattr(config, "auto_map", {})
    class_ref = auto_map.get("AutoModel")
    if not class_ref:
        return None

    cls = get_class_from_dynamic_module(class_ref, MODEL_PATH, trust_remote_code=True)

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
        logger.info("Pre-patched MossTTSDelayModel.get_input_embeddings")

    return cls


@serve.deployment(
    name="moss_soundeffect",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0},
)
class MossSoundEffectDeployment(BaseGPUDeployment):
    """MOSS-SoundEffect text-to-sound via native PyTorch inference."""

    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = None

    def _load(self, model_name: str = "moss-soundeffect") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"MOSS-SoundEffect model not found at {MODEL_PATH}")

        _patch_transformers()
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Pre-patch the model class before from_pretrained instantiates it
        _get_moss_model_class()

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Load processor manually to avoid transformers 5.x type-checking
        # the audio_tokenizer against AutoModel (MossTTSDelayModel != AutoModel)
        logger.info("Loading MOSS processor components from %s", MODEL_PATH)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )

        # Audio tokenizer is a separate 3.5B model. Keep on CPU to save VRAM
        # for the 8B main model (~16GB on 24GB card).
        codec_path = os.environ.get(
            "MOSS_AUDIO_TOKENIZER_PATH",
            os.path.join(MODELS_ROOT, "audio/moss-audio-tokenizer"),
        )
        logger.info("Loading MOSS audio tokenizer from %s (CPU)", codec_path)
        audio_tokenizer = AutoModel.from_pretrained(
            codec_path, trust_remote_code=True, local_files_only=True,
            device_map="cpu", torch_dtype=torch.float32,
        )

        # Get the processor class from the model's custom code.
        # Construct manually — bypass super().__init__() because transformers 5.x
        # ProcessorMixin insists on feature_extractor + type-checks audio_tokenizer
        # against AutoModel (MossAudioTokenizerModel doesn't match).
        proc_cls = get_class_from_dynamic_module(
            "processing_moss_tts.MossTTSDelayProcessor",
            MODEL_PATH, trust_remote_code=True,
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

        # Model config has pad_token_id=None — set it from the tokenizer
        # so the processor's _pad can assign it during batching
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

        processor.audio_user_slot_token = _id_to_token(
            config.audio_user_slot_token_id
        )
        processor.audio_assistant_gen_slot_token = _id_to_token(
            config.audio_assistant_gen_slot_token_id
        )
        processor.audio_assistant_delay_slot_token = _id_to_token(
            config.audio_assistant_delay_slot_token_id
        )
        processor.audio_start_token = _id_to_token(config.audio_start_token_id)
        processor.audio_end_token = _id_to_token(config.audio_end_token_id)
        self.processor = processor

        # MOSS's get_input_embeddings(input_ids) requires a positional arg,
        # but transformers 5.x tie_weights() calls it without args during loading.
        # Skip weight tying — MOSS has multi-head output so tying doesn't apply.
        config.tie_word_embeddings = False

        logger.info("Loading MOSS model (%s) from %s", dtype, MODEL_PATH)
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModel.from_pretrained(
            MODEL_PATH,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": 0} if self.device == "cuda" else None,
        )
        model.eval()

        self.model = model
        self.model_name = model_name

        vram = torch.cuda.memory_allocated(0) / (1024**2) if self.device == "cuda" else 0
        logger.info("MOSS-SoundEffect loaded on %s (VRAM: %.0fMB)", self.device, vram)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        super()._unload()

    def _generate_audio(self, prompt: str, tokens=None) -> bytes:
        batch_spec = {}
        if tokens is not None:
            batch_spec["tokens"] = tokens

        conversations = [
            [self.processor.build_user_message(ambient_sound=prompt, **batch_spec)]
        ]
        batch = self.processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )

        results = self.processor.decode(outputs)
        if not results:
            raise RuntimeError("No audio generated")

        audio = results[0].audio_codes_list[0]
        buf = io.BytesIO()
        import torchaudio
        sample_rate = self.processor.model_config.sampling_rate
        torchaudio.save(buf, audio.unsqueeze(0), sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {prompt, tokens}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        import asyncio

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "moss-soundeffect")

            prompt = extracted.get("prompt", "")
            if not prompt:
                return JSONResponse(self.handle_error("prompt required"), status_code=400)

            audio = await asyncio.to_thread(self._generate_audio, prompt, extracted.get("tokens"))

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("moss_soundeffect error: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)
