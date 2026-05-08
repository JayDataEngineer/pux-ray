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

MODEL_PATH = os.environ.get("MOSS_SFX_MODEL_PATH", "/models/audio/moss-soundeffect")


def _patch_transformers():
    """Apply all transformers compat patches needed by MOSS."""
    from services.compat import apply
    apply()

    import transformers.processing_utils as _pu
    if not hasattr(_pu, "MODALITY_TO_BASE_CLASS_MAPPING"):
        _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}


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

        from transformers import AutoModel, AutoProcessor

        logger.info("Loading MOSS processor from %s", MODEL_PATH)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            codec_path=MODEL_PATH,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)

        logger.info("Loading MOSS model (%s) from %s", dtype, MODEL_PATH)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
            device_map=self.device,
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
