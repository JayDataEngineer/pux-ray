"""MOSS-SoundEffect — Text-to-sound effect generation.

8B parameter model from the MOSS-TTS family. Generates environmental sounds,
urban scenes, creatures, human actions, and music-like clips from text prompts.
Requires ~22GB VRAM.
"""
from __future__ import annotations

import io
import logging
import os

import torch
from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment, _free_cuda_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MOSS_SFX_MODEL_PATH", "/models/audio/moss-soundeffect")


@serve.deployment(
    name="moss_soundeffect",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class MossSoundEffectDeployment(BaseGPUDeployment):
    """MOSS-SoundEffect text-to-sound."""

    def _load(self, model_name: str = "moss-soundeffect") -> None:
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

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
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

        self.device = device
        self.model_name = model_name
        logger.info("MOSS-SoundEffect loaded from %s on %s", MODEL_PATH, device)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        _free_cuda_cache()

    async def __call__(self, request):
        if not self.is_loaded():
            self.load_model("moss-soundeffect")

        import torchaudio

        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        tokens = body.get("tokens")
        batch_spec = {}
        if tokens is not None:
            batch_spec["tokens"] = tokens

        try:
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
                return JSONResponse({"error": "no audio generated"}, status_code=500)

            audio = results[0].audio_codes_list[0]
            buf = io.BytesIO()
            sample_rate = self.processor.model_config.sampling_rate
            torchaudio.save(buf, audio.unsqueeze(0), sample_rate, format="WAV")
            buf.seek(0)
            return Response(content=buf.read(), media_type="audio/wav")

        except Exception as e:
            logger.exception("MOSS-SoundEffect generation failed")
            return JSONResponse({"error": str(e)}, status_code=500)
