"""MOSS-SoundEffect — Text-to-sound effect generation.

8B parameter model from the MOSS-TTS family. Generates environmental sounds,
urban scenes, creatures, human actions, and music-like clips from text prompts.
Requires ~22GB VRAM.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

# Apply compat patches at module import time, before any model code loads
from services.compat import apply as _apply_compat
_apply_compat()

import asyncio
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


@serve.deployment(
    name="moss_soundeffect",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
            },
        },
    },
)
class MossSoundEffectDeployment(BaseGPUDeployment):
    """MOSS-SoundEffect text-to-sound."""

    def _load(self, model_name: str = "moss-soundeffect") -> None:
        from services.compat import apply
        apply()

        import transformers.processing_utils as _pu
        if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
            _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}

        import importlib.util

        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"MOSS-SoundEffect model not found at {MODEL_PATH}")

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        if (
            device == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability()
            attn_impl = "flash_attention_2" if major >= 8 else "sdpa"
        else:
            attn_impl = "sdpa" if device == "cuda" else "eager"

        from transformers import AutoModel, AutoProcessor

        # Pre-load the model class to patch get_input_embeddings before
        # AutoModel.from_pretrained() instantiates it.
        # The MOSS model's get_input_embeddings(input_ids) takes an argument but
        # transformers.generate() calls it without args to get the nn.Embedding layer.
        import importlib
        try:
            modeling = importlib.import_module("modeling_moss_tts")
            _orig = getattr(modeling, 'MossTTSDelayModel', None)
            if _orig and callable(getattr(_orig, 'get_input_embeddings', None)):
                _orig_get_emb = _orig.get_input_embeddings
                def _compat_get_emb(self_inner, input_ids=None):
                    if input_ids is None:
                        return self_inner.language_model.get_input_embeddings()
                    return _orig_get_emb(self_inner, input_ids)
                _orig.get_input_embeddings = _compat_get_emb
        except ImportError:
            pass  # trust_remote_code will load it

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            codec_path=MODEL_PATH,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device)

        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        self.model.eval()

        # Patch get_input_embeddings: MOSS model expects (self, input_ids) but
        # transformers.generate() calls it with (self) to get the nn.Embedding.
        self._patch_get_input_embeddings()

        self.device = device
        self.model_name = model_name
        logger.info("MOSS-SoundEffect loaded from %s on %s", MODEL_PATH, device)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        super()._unload()

    def _patch_get_input_embeddings(self):
        """Patch get_input_embeddings to be callable without arguments.

        The MOSS model's get_input_embeddings(self, input_ids) expects an argument,
        but transformers.generate() calls it with no args to get the nn.Embedding.
        """
        import inspect
        method = type(self.model).get_input_embeddings
        sig = inspect.signature(method)
        # Count required params (excluding self)
        required = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        if len(required) > 1:  # self + at least one required arg
            _orig = method
            def _compat(self_inner, input_ids=None):
                if input_ids is None:
                    # Return the underlying embedding layer
                    lm = getattr(self_inner, 'language_model', self_inner)
                    return lm.get_input_embeddings() if hasattr(lm, 'get_input_embeddings') else lm
                return _orig(self_inner, input_ids)
            type(self.model).get_input_embeddings = _compat
            logger.info("Patched get_input_embeddings for MOSS model")

    def _generate_audio(self, prompt: str, tokens) -> bytes:
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
            raise RuntimeError("no audio generated")

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
            logger.error("moss_soundeffect error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)