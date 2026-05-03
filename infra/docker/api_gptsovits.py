"""GPT-SoVITS API server — voice cloning TTS inside Docker."""
from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import uvicorn

app = FastAPI(title="GPT-SoVITS API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    refer_wav: UploadFile = File(...),
    prompt_text: str = Form(""),
    prompt_language: str = Form("en"),
    text_language: str = Form("en"),
):
    """Voice-cloned TTS. Takes text + reference audio, returns synthesized WAV."""
    if not text:
        return Response(content=b'{"error":"text is required"}', media_type="application/json", status_code=400)

    ref_bytes = await refer_wav.read()
    tmpdir = tempfile.mkdtemp(prefix="gptsovits_")
    try:
        refer_path = Path(tmpdir) / "reference.wav"
        refer_path.write_bytes(ref_bytes)
        output_path = Path(tmpdir) / "output.wav"

        import subprocess
        result = subprocess.run(
            [
                "python", "api_v2.py",
                "--text", text,
                "--text_language", text_language,
                "--refer_wav", str(refer_path),
                "--prompt_text", prompt_text,
                "--prompt_language", prompt_language,
                "--output", str(output_path),
            ],
            capture_output=True, text=True, timeout=300,
            cwd="/opt/gpt-sovits",
        )
        if result.returncode != 0:
            raise RuntimeError(f"GPT-SoVITS failed: {result.stderr[-500:]}")

        if not output_path.exists():
            raise RuntimeError("GPT-SoVITS did not produce audio output")

        return Response(content=output_path.read_bytes(), media_type="audio/wav")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
