"""IndexTTS - GPU text-to-speech using IndexTTS-2 model.

High-quality multi-speaker TTS. Requires ~13GB VRAM.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("INDEXTTS_MODEL_PATH", "/models/tts/index-tts")


@serve.deployment(
    name="index_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
    },
)
class IndexTTSDeployment(BaseGPUDeployment):
    """GPU-based IndexTTS-2. High quality multi-speaker TTS."""

    def _load(self, model_name: str = "index-tts") -> None:
        from services.compat import apply
        apply()

        model_dir = MODEL_PATH
        cfg_path = os.path.join(model_dir, "config.yaml")

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"IndexTTS config not found at {cfg_path}. "
                f"Run model-sync to download it."
            )

        from indextts.infer_v2 import IndexTTS2
        self.model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,
            device="cuda",
        )
        self.model_name = model_name
        logger.info("IndexTTS loaded from %s", model_dir)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        super()._unload()

    def synthesize(
        self,
        text: str,
        voice: str = "default",
    ) -> bytes:
        """Synthesize speech. Returns audio bytes."""
        import tempfile
        import soundfile as sf

        if os.path.isfile(voice):
            spk_prompt = voice
        else:
            default_ref = "/models/tts/kokoro/samples/af_heart_0.wav"
            if os.path.isfile(default_ref):
                spk_prompt = default_ref
            else:
                import numpy as np
                prompt = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                prompt_path = prompt.name
                tone = np.sin(2 * np.pi * 440 * np.arange(0, 1.0, 1 / 24000)).astype(np.float32)
                sf.write(prompt_path, tone, 24000)
                spk_prompt = prompt_path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            self.model.infer(
                spk_audio_prompt=spk_prompt,
                text=text,
                output_path=output_path,
            )
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            for p in [spk_prompt, output_path]:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, voice}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "index-tts")

            audio = await asyncio.to_thread(
                lambda: self.synthesize(
                    text=extracted.get("text", ""),
                    voice=extracted.get("voice", "default"),
                ),
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("index_tts error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)