"""Tech Noir Ray SDK - async HTTP client for the Ray AI infrastructure."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union

import httpx


class RayClient:
    """Async client for the Tech Noir Ray infrastructure.

    Usage:
        client = RayClient()
        reply = await client.chat("What is 2+2?")
        audio = await client.synthesize("Hello world")
        result = await client.transcribe("meeting.wav")
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def chat(
        self,
        messages: Union[str, list[dict[str, str]]],
        model: str = "qwen3.6-27b-iq4_nl",
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Chat with an LLM. Auto-swaps GPU model if needed."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        resp = await self.client.post(
            "/v1/chat/completions",
            json={"messages": messages, "model": model, "stream": stream, **kwargs},
        )
        resp.raise_for_status()
        return resp.json()

    async def transcribe(
        self,
        audio: Union[str, Path, bytes],
        model: str = "faster-whisper",
        **kwargs,
    ) -> dict[str, Any]:
        """Transcribe audio. Auto-loads ASR model."""
        if isinstance(audio, (str, Path)):
            with open(audio, "rb") as f:
                audio_bytes = f.read()
            filename = Path(audio).name
        elif isinstance(audio, bytes):
            audio_bytes = audio
            filename = "audio.wav"
        else:
            raise TypeError(f"audio must be str, Path, or bytes, got {type(audio)}")

        resp = await self.client.post(
            "/v1/audio/transcriptions",
            files={"file": (filename, io.BytesIO(audio_bytes))},
            data={"model": model, **{k: str(v) for k, v in kwargs.items()}},
        )
        resp.raise_for_status()
        return resp.json()

    async def synthesize(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_bella",
        **kwargs,
    ) -> bytes:
        """Text-to-speech. Returns audio bytes."""
        resp = await self.client.post(
            "/v1/audio/speech",
            json={"input": text, "model": model, "voice": voice, **kwargs},
        )
        resp.raise_for_status()
        return resp.content

    async def synthesize_to_file(
        self,
        text: str,
        path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """Synthesize and save to file."""
        audio = await self.synthesize(text, **kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return path

    async def generate_3d(
        self,
        image: Union[str, Path, bytes],
        model: str = "trellis",
        **kwargs,
    ) -> bytes:
        """Generate 3D mesh from image. Returns GLB bytes."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
            filename = Path(image).name
        elif isinstance(image, bytes):
            image_bytes = image
            filename = "image.png"
        else:
            raise TypeError(f"image must be str, Path, or bytes, got {type(image)}")

        resp = await self.client.post(
            "/v1/3d/generate",
            files={"image": (filename, io.BytesIO(image_bytes))},
            data={"model": model, **{k: str(v) for k, v in kwargs.items()}},
        )
        resp.raise_for_status()
        return resp.content

    async def status(self) -> dict[str, Any]:
        """Get infrastructure status (GPU, loaded models, VRAM)."""
        resp = await self.client.get("/status")
        resp.raise_for_status()
        return resp.json()

    async def load(self, service: str, model: str) -> dict[str, Any]:
        """Explicitly load a model."""
        resp = await self.client.post(
            "/admin/load", json={"service": service, "model": model},
        )
        resp.raise_for_status()
        return resp.json()

    async def unload(self) -> dict[str, Any]:
        """Unload all GPU models."""
        resp = await self.client.post("/admin/unload")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
