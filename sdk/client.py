"""Tech Noir Ray SDK - async HTTP client for the Ray AI infrastructure.

All services use the unified TNAP protocol via /v1/{service}/generate.
Legacy methods (chat, synthesize, transcribe) remain for convenience.
"""
from __future__ import annotations

import base64
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
        result = await client.generate("kokoro", input={"text": "hello"})
    """

    def __init__(self, base_url: str = "http://localhost:18080", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    # ── Generic TNAP generate ──────────────────────────────────────────────────

    async def generate(
        self,
        service: str,
        input: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Generate via any service using the TNAP protocol.

        Args:
            service: Service name from the registry (e.g. "kokoro", "trellis").
            input: TNAP input dict with text, prompt, image_b64, audio_b64, etc.
            config: InferenceConfig overrides (precision, quantization, low_resource).
            **params: Additional top-level params.

        Returns:
            TNAPResponse dict with status, output, metrics.
        """
        payload: dict[str, Any] = {
            "action": "generate",
            "input": input or {},
        }
        if config:
            payload["config"] = config
        payload.update(params)

        resp = await self.client.post(
            f"/v1/{service}/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def generate_binary(
        self,
        service: str,
        input: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate and return the decoded binary output from a TNAP service.

        Decodes the base64 content from the TNAP response output.
        """
        result = await self.generate(service, input=input, **kwargs)
        output = result.get("output", {})
        content_b64 = output.get("content", "")
        if not content_b64:
            raise ValueError(f"No binary output from {service}")
        return base64.b64decode(content_b64)

    # ── Service discovery ──────────────────────────────────────────────────────

    async def services(self) -> list[dict[str, Any]]:
        """List all registered services."""
        resp = await self.client.get("/v1/services")
        resp.raise_for_status()
        return resp.json()

    async def service_info(self, service: str) -> dict[str, Any]:
        """Get info about a specific service."""
        resp = await self.client.get(f"/v1/services/{service}")
        resp.raise_for_status()
        return resp.json()

    # ── Convenience wrappers ───────────────────────────────────────────────────

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

    # ── Infrastructure ─────────────────────────────────────────────────────────

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
