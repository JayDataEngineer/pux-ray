"""VibeVoice TTS handler for Docker worker.

Wraps VibeVoice 7B long-form multi-speaker TTS as an HTTP endpoint.
Model stays loaded in GPU memory between requests.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger("workers.vibevoice")


class Handler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name: str | None = None

    async def health(self):
        if self.model is None:
            return {"status": "model_not_loaded"}
        return {"status": "ok", "model": self.model_name}

    async def load(self, body: dict):
        """Load VibeVoice model into GPU memory."""
        if self.model is not None:
            return {"status": "already_loaded", "model": self.model_name}

        model_path = os.environ.get("MODEL_PATH", "")
        if not model_path:
            return {"status": "error", "message": "MODEL_PATH not set"}
        logger.info("Loading VibeVoice model from %s", model_path)

        import torch
        from transformers import AutoProcessor
        from vibevoice import VibeVoiceForConditionalGenerationInference

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model_name = model_path

        logger.info("VibeVoice loaded successfully")
        return {"status": "loaded", "model": self.model_name}

    async def generate(self, request: Request):
        """Synthesize speech from text.

        Accepts application/json with:
          - input: text string (required). Use "Speaker 1: ...\nSpeaker 2: ..."
                   for multi-speaker.
          - speaker_names: list of voice names (default: ["Andrew"])
          - output_format: "wav" (default)
        """
        if self.model is None:
            await self.load({})

        body = await request.json()
        text = body.get("input", "")
        if not text:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = speaker_names.split(",")

        # Normalize: wrap plain text as single-speaker
        if "Speaker " not in text and "speaker " not in text:
            text = f"Speaker 1: {text}"

        import torch
        import soundfile as sf

        logger.info("Synthesizing speech (%d chars, speakers: %s)", len(text), speaker_names)

        with tempfile.TemporaryDirectory(prefix="vibevoice_") as tmpdir:
            # Write input text file
            txt_path = Path(tmpdir) / "input.txt"
            txt_path.write_text(text)

            # VibeVoice inference
            # Build voice prompts from speaker names
            repo_root = os.environ.get("REPO_ROOT", "/app/repo")
            voices_dir = Path(repo_root) / "voices"

            inputs = self.processor(
                text=text,
                speaker_names=speaker_names,
                voices_dir=str(voices_dir),
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                audio = self.model.generate(**inputs)

            # audio is a tensor — convert to WAV bytes
            audio_np = audio.cpu().numpy().squeeze()
            output_path = Path(tmpdir) / "output.wav"
            sf.write(str(output_path), audio_np, samplerate=24000)

            wav_bytes = output_path.read_bytes()
            logger.info("Generated audio (%d bytes, %.1fs)",
                       len(wav_bytes), len(audio_np) / 24000)

            return Response(content=wav_bytes, media_type="audio/wav")
