"""Forge adapter for ACE-Step C++ music generation.

Source: github.com/ServeurpersoCom/acestep.cpp (upstream, unmodified)
Docker: forge-reg.local:30500/tech-noir/ace-step:latest (Dockerfile.acetep)
Pool:   inference-ace-step, port 8056

Two-step pipeline:
  1. POST /lm   — generate music codes from caption + (optional) lyrics
  2. POST /synth — render codes to 48kHz stereo WAV bytes

Turbo vs Base is selected at request time by specifying the DiT GGUF filename:
  turbo: acestep-v15-turbo-Q8_0  (8-step, fast)
  base:  acestep-v15-sft-Q8_0    (50-step, higher quality)

Both DiT GGUFs live in the same models directory — no container restart required.
"""
from __future__ import annotations

import base64
import logging
import os
import time

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

ACE_STEP_URL = os.environ.get("ACE_STEP_URL", "http://ace-step-service:8080")

# GGUF model filenames inside /models/audio/acestep-cpp/
_DIT_MODELS = {
    "turbo": "acestep-v15-turbo-Q8_0",
    "base":  "acestep-v15-sft-Q8_0",
}
_LM_MODEL      = "acestep-5Hz-lm-1.7B-Q8_0"
_TEXT_ENCODER  = "Qwen3-Embedding-0.6B-Q8_0"
_VAE_MODEL     = "vae-BF16"


class ACEStepForgeService(ForgeService):
    """Calls acestep.cpp server via two-step HTTP API.

    Step 1  POST /lm   — text → music codes (LM pass)
    Step 2  POST /synth — codes → WAV bytes (DiT render pass)

    Variant (turbo vs base) is controlled by the ``variant`` key in the
    infer payload. Defaults to "turbo".

    VRAM: ~8GB — shares GPU with MOSS/diarization/llama.cpp.
    """

    service_name = "ace-step"
    default_model = "ace-step-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0  # managed by the inference pool, not the forge

    def __init__(self):
        super().__init__()
        self._server_url: str | None = None

    def _find_server(self) -> str | None:
        candidates = [
            ACE_STEP_URL,
            "http://localhost:8056",
            "http://127.0.0.1:8056",
        ]
        for url in candidates:
            try:
                with httpx.Client(timeout=5) as c:
                    if c.get(f"{url}/health").status_code == 200:
                        return url
            except Exception:
                continue
        return None

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        url = self._find_server()
        if url:
            self._server_url = url
            logger.info("ACE-Step: server reachable at %s", url)
        else:
            logger.warning("ACE-Step: server not reachable — will retry on first request")
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self._server_url = None

    def infer(self, payload: dict) -> dict:
        """Generate music via ACE-Step C++ two-step pipeline.

        payload keys:
          prompt / caption  (str, required) — musical description
          lyrics            (str, optional) — lyrics text
          variant           ("turbo" | "base", default: "turbo")
          duration_s        (float, default: 30.0) — target duration in seconds
          temperature       (float, default: 1.0)
          top_k             (int, default: 250)
          top_p             (float, default: 0.0)
          seed              (int, optional)

        Returns:
          {"status": "success", "output": {"type": "audio", "audio_b64": "..."}}
        """
        url = self._server_url or self._find_server()
        if not url:
            return {"status": "error", "error": "ACE-Step server not reachable"}
        self._server_url = url

        caption = payload.get("prompt") or payload.get("caption", "")
        if not caption:
            return {"status": "error", "error": "prompt / caption is required"}

        variant = payload.get("variant", "turbo")
        if variant not in _DIT_MODELS:
            variant = "turbo"
        dit_model = _DIT_MODELS[variant]

        lm_body = {
            "caption": caption,
            "lm_model": _LM_MODEL,
            "dit_model": dit_model,
            "text_encoder": _TEXT_ENCODER,
        }
        if payload.get("lyrics"):
            lm_body["lyrics"] = payload["lyrics"]
        for key in ("temperature", "top_k", "top_p", "seed"):
            if key in payload:
                lm_body[key] = payload[key]

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=600) as client:
                # Step 1: LM pass — generate music codes
                lm_resp = client.post(f"{url}/lm", json=lm_body)
                if lm_resp.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"ACE-Step /lm failed {lm_resp.status_code}: {lm_resp.text[:200]}",
                    }
                music_codes = lm_resp.json()

                # Step 2: Synth pass — render codes to audio
                synth_body = {
                    **music_codes,
                    "vae_model": _VAE_MODEL,
                }
                if payload.get("duration_s"):
                    synth_body["duration"] = payload["duration_s"]

                synth_resp = client.post(f"{url}/synth", json=synth_body)
                if synth_resp.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"ACE-Step /synth failed {synth_resp.status_code}: {synth_resp.text[:200]}",
                    }

        except Exception as e:
            return {"status": "error", "error": f"ACE-Step request failed: {e}"}

        elapsed = time.perf_counter() - t0

        # synth returns raw WAV bytes
        audio_b64 = base64.b64encode(synth_resp.content).decode()

        return {
            "status": "success",
            "output": {
                "type": "audio",
                "format": "wav",
                "audio_b64": audio_b64,
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "variant": variant,
                "dit_model": dit_model,
            },
        }

    def actual_vram_mb(self) -> int:
        return 0
