"""Kokoro TTS API server — sherpa-onnx backend.

Replaces the previous native PyTorch Kokoro implementation (5500+ lines across
opt/wan2gp/preprocessing/kokoro, opt/wan2gp/models/kokoro, vendor/wan2gp/*, and
services/wan2gp/custom_models/kokoro). This single file is the entire TTS engine.

OpenAI-compatible endpoints:
  GET  /health
  GET  /v1/audio/voices           → list of supported voice names
  POST /v1/audio/speech           → OpenAI TTS (JSON in, audio/wav out)
  POST /synthesize                → legacy alias for /v1/audio/speech

Model: kokoro-multi-lang-v1_0 (53 voices, English + Chinese, 24 kHz mono).
Voices match the upstream hexgrad/Kokoro-82M names exactly (af_bella, am_adam,
zf_xiaobei, ...) so the MCP tool's KOKORO_VOICES catalog stays authoritative.
"""
from __future__ import annotations

import io
import logging
import os
import struct
import wave
from typing import Any

import sherpa_onnx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kokoro")

MODEL_DIR = os.environ.get("KOKORO_MODEL_DIR", "/models/tts/kokoro-sherpa")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices.bin")
TOKENS_PATH = os.path.join(MODEL_DIR, "tokens.txt")
DATA_DIR = os.path.join(MODEL_DIR, "espeak-ng-data")
LEXICON_EN = os.path.join(MODEL_DIR, "lexicon-us-en.txt")
LEXICON_GB = os.path.join(MODEL_DIR, "lexicon-gb-en.txt")
LEXICON_ZH = os.path.join(MODEL_DIR, "lexicon-zh.txt")
RULE_FSTS = ",".join(filter(os.path.exists, (
    os.path.join(MODEL_DIR, "date-zh.fst"),
    os.path.join(MODEL_DIR, "number-zh.fst"),
    os.path.join(MODEL_DIR, "phone-zh.fst"),
)))

# Voice name → speaker ID map for kokoro-multi-lang-v1_0 (53 voices, EN+ZH).
# The sherpa-onnx Python wrapper (1.12+) does not expose name2id even though
# the C++ binary does — embed the map directly. Source: k2-fsa.github.io/sherpa/
# onnx/tts/pretrained_models/kokoro.html (kokoro-multi-lang-v1_0).
# Must match mcp/wan2gp-studio/src/tools/tts.py KOKORO_VOICES.
VOICE_NAME_TO_ID: dict[str, int] = {
    "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4,
    "af_kore": 5, "af_nicole": 6, "af_nova": 7, "af_river": 8, "af_sarah": 9,
    "af_sky": 10, "am_adam": 11, "am_echo": 12, "am_eric": 13, "am_fenrir": 14,
    "am_liam": 15, "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19,
    "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23,
    "bm_daniel": 24, "bm_fable": 25, "bm_george": 26, "bm_lewis": 27,
    "ef_dora": 28, "em_alex": 29, "ff_siwis": 30,
    "hf_alpha": 31, "hf_beta": 32, "hm_omega": 33, "hm_psi": 34,
    "if_sara": 35, "im_nicola": 36,
    "jf_alpha": 37, "jf_gongitsune": 38, "jf_nezumi": 39, "jf_tebukuro": 40,
    "jm_kumo": 41, "pf_dora": 42, "pm_alex": 43, "pm_santa": 44,
    "zf_xiaobei": 45, "zf_xiaoni": 46, "zf_xiaoxiao": 47, "zf_xiaoyi": 48,
    "zm_yunjian": 49, "zm_yunxi": 50, "zm_yunxia": 51, "zm_yunyang": 52,
}

app = FastAPI(title="Kokoro TTS (sherpa-onnx)", version="1.0")
_tts: sherpa_onnx.OfflineTts | None = None
_LOAD_MS: int = 0


def _build_lexicon() -> str:
    avail = [p for p in (LEXICON_EN, LEXICON_GB, LEXICON_ZH) if os.path.exists(p)]
    return ",".join(avail)


@app.on_event("startup")
def load_model() -> None:
    """Load sherpa-onnx Kokoro once at startup."""
    global _tts, _LOAD_MS
    import time
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Kokoro ONNX model not found at {MODEL_PATH}. "
            f"Download kokoro-multi-lang-v1_0 from "
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
            f"kokoro-multi-lang-v1_0.tar.bz2 and extract into {MODEL_DIR}/"
        )
    t0 = time.time()
    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=MODEL_PATH,
        voices=VOICES_PATH,
        tokens=TOKENS_PATH,
        lexicon=_build_lexicon(),
        data_dir=DATA_DIR,
    )
    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        kokoro=kokoro,
        num_threads=int(os.environ.get("KOKORO_THREADS", "2")),
        debug=False,
        provider=os.environ.get("KOKORO_PROVIDER", "cpu"),
    )
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=model_cfg,
        rule_fsts=RULE_FSTS,
        max_num_sentences=1,
        silence_scale=0.2,
    )
    _tts = sherpa_onnx.OfflineTts(cfg)
    _LOAD_MS = int((time.time() - t0) * 1000)
    logger.info(
        "Kokoro loaded in %d ms — %d voices, sample_rate=%d",
        _LOAD_MS, _tts.num_speakers, _tts.sample_rate,
    )


def _wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    """Convert float32 samples [-1, 1] → 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        ))
    return buf.getvalue()


def _resolve_sid(voice: str | None) -> int:
    """Map voice name → speaker ID. Default to af_bella (ID 2)."""
    if not voice:
        return VOICE_NAME_TO_ID.get("af_bella", 0)
    if voice.isdigit():
        return int(voice)
    sid = VOICE_NAME_TO_ID.get(voice)
    if sid is None:
        raise ValueError(
            f"Unknown voice '{voice}'. Available: {sorted(VOICE_NAME_TO_ID)[:10]} ... "
            f"({len(VOICE_NAME_TO_ID)} total). GET /v1/audio/voices for full list."
        )
    return sid


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _tts is not None else "loading",
        "engine": "sherpa-onnx",
        "model": "kokoro-multi-lang-v1_0",
        "voices": len(VOICE_NAME_TO_ID),
        "load_time_ms": _LOAD_MS,
        "sample_rate": _tts.sample_rate if _tts else 24000,
    }


@app.get("/v1/audio/voices")
def list_voices() -> dict[str, Any]:
    """List all available voices (alphabetical)."""
    return {
        "voices": sorted(VOICE_NAME_TO_ID.keys()),
        "count": len(VOICE_NAME_TO_ID),
        "default": "af_bella",
    }


@app.post("/v1/audio/speech")
@app.post("/synthesize")
async def synthesize(body: dict[str, Any]) -> Response:
    """OpenAI-compatible TTS endpoint.

    Request JSON:
      {
        "input": "Hello world",        # required — text to synthesize
        "voice": "af_bella",           # optional — name or numeric ID
        "speed": 1.0,                  # optional — 0.5–2.0, default 1.0
        "response_format": "wav"       # only wav supported (PCM 24kHz mono)
      }
    """
    if _tts is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    text = (body.get("input") or body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "input (text) is required"}, status_code=400)

    try:
        sid = _resolve_sid(body.get("voice"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    speed = float(body.get("speed", 1.0))
    if not 0.25 <= speed <= 4.0:
        return JSONResponse({"error": "speed must be in [0.25, 4.0]"}, status_code=400)

    import time
    t0 = time.time()
    audio = _tts.generate(text=text, sid=sid, speed=speed)
    inf_ms = int((time.time() - t0) * 1000)

    pcm = _wav_bytes(audio.samples, audio.sample_rate)
    headers = {
        "X-Inference-Time-s": f"{inf_ms / 1000:.3f}",
        "X-Audio-Duration-s": f"{len(audio.samples) / audio.sample_rate:.3f}",
        "X-RTF": f"{inf_ms / 1000 / (len(audio.samples) / audio.sample_rate):.3f}",
        "X-Sample-Rate": str(audio.sample_rate),
        "X-Voice": body.get("voice") or "af_bella",
    }
    return Response(content=pcm, media_type="audio/wav", headers=headers)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "kokoro",
        "engine": "sherpa-onnx",
        "endpoints": "/health, /v1/audio/voices, /v1/audio/speech, /synthesize",
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("KOKORO_PORT", "8060")),
        log_level=os.environ.get("KOKORO_LOG_LEVEL", "info"),
    )
