"""Qwen3-TTS API server — multi-speaker TTS with CustomVoice inside Docker."""
from __future__ import annotations

import os
import io
import logging

import torch
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import uvicorn

app = FastAPI(title="Qwen3-TTS API")

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/tts/qwen3-tts-12hz-1.7b-customvoice")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/models/tts/qwen3-tts-tokenizer-12hz")

_model = None
logger = logging.getLogger(__name__)


@app.on_event("startup")
def load_model():
    global _model
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Qwen3-TTS model not found at {MODEL_PATH} — mount model volume")
    from qwen_tts import Qwen3TTSModel
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    logger.info("Qwen3-TTS loaded from %s", MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _model is not None}


@app.post("/synthesize")
async def synthesize(data: dict):
    text = data.get("input", "")
    if not text:
        return JSONResponse({"error": "input text is required"}, status_code=400)

    voice = data.get("voice", "Aiden")
    instruct = data.get("instruct", "")

    zh_speakers = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"}
    ja_speakers = {"Ono_Anna"}
    ko_speakers = {"Sohee"}

    if voice in zh_speakers:
        lang = "Chinese"
    elif voice in ja_speakers:
        lang = "Japanese"
    elif voice in ko_speakers:
        lang = "Korean"
    else:
        lang = "English"

    gen_kwargs = {}
    if instruct:
        gen_kwargs["instruct"] = instruct

    wavs, sr = _model.generate_custom_voice(
        text=text,
        language=lang,
        speaker=voice,
        **gen_kwargs,
    )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
