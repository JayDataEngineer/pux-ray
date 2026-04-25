"""GPT-SoVITS - voice cloning TTS.

Clones voices from reference audio. GPU required.
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0.01,
        "runtime_env": {
            "pip": ["torch>=2.1", "torchaudio", "transformers>=4.40",
                    "librosa", "soundfile", "numpy", "scipy"],
        },
    },
)
class GPTSoVITSDeployment(BaseGPUDeployment):
    """GPT-SoVITS voice cloning TTS."""

    def _load(self, model_name: str = "gpt-sovits") -> None:
        from registry.models import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.get_path("tts", model_name)
        # GPT-SoVITS has its own model loading from GPT_SoVITS/ directory
        # The actual import comes from the installed gptsovits package
        logger.info("GPT-SoVITS loaded from %s (stub - needs model code)", model_path)
        self.model = True  # placeholder
        self.model_name = model_name

    def _unload(self) -> None:
        self.model = None

    async def synthesize(self, text: str, refer_wav: bytes = b"",
                         prompt_text: str = "", prompt_language: str = "en",
                         text_language: str = "en") -> bytes:
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        # GPT-SoVITS inference - needs reference audio for voice cloning
        raise NotImplementedError("GPT-SoVITS inference TBD - needs model code")

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(text=body.get("input", ""))
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
