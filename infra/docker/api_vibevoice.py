"""VibeVoice API server — long-form multi-speaker TTS inside Docker."""
from __future__ import annotations

import os
import io
import sys

sys.path.insert(0, "/app/repo")

import torch
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import uvicorn

app = FastAPI(title="VibeVoice API")

_pipeline = None
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/audio/vibevoice/VibeVoice-7B")


@app.on_event("startup")
def load_model():
    global _pipeline
    from vibevoice.model import VibeVoicePipeline
    _pipeline = VibeVoicePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    print(f"VibeVoice loaded: {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _pipeline is not None}


@app.post("/generate")
async def generate(data: dict):
    """Generate long-form speech. Returns WAV bytes."""
    text = data.get("input", "")
    speaker_names = data.get("speaker_names", ["Andrew"])
    if isinstance(speaker_names, str):
        speaker_names = [s.strip() for s in speaker_names.split(",")]

    if not text:
        return JSONResponse({"error": "input text is required"}, status_code=400)

    output = _pipeline.run(text=text, speaker_names=speaker_names)

    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, output["audio"], output["sample_rate"], format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
