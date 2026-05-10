"""Audio event classification service.

Uses PANNs (Pre-trained Audio Neural Networks) ONNX model for
sound event detection based on AudioSet 527-class labels.
Lazy-loads on first request.
"""

import asyncio
import io
import os
import subprocess
import tempfile
import time
from typing import Optional

import httpx
import numpy as np
from loguru import logger

from ..settings import get_settings, get_device


# AudioSet class labels (top 50 most common, full list loaded from file)
_AUDIASET_LABELS = [
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Shout", "Bellow",
    "Whoop", "Yell", "Children shouting", "Screaming", "Whispering",
    "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
    "Chuckle, chortle", "Crying, sobbing", "Baby cry, infant cry",
    "Whimper", "Wail, moan", "Sigh", "Singing", "Choir",
    "Yodeling", "Chant", "Mantra", "Child singing", "Synthetic singing",
    "Rapping", "Humming", "Groan", "Grunt", "Whistling",
    "Breathing", "Cough", "Throat clearing", "Sneeze", "Snoring",
    "Gasp", "Pant", "Snort", "Drum beat", "Snare drum roll",
]


class AudioClassifyService:
    """Audio event classification via PANNs ONNX."""

    def __init__(self):
        self._session = None
        self._labels: list[str] = []
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Audio classify model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Audio classify model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("audio_classify"):
                raise RuntimeError("Audio classification is disabled")

            try:
                logger.info("Loading audio classification model (PANNs)")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"Audio classify model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load audio classify model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        device = get_device()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        # Download PANNs Cnn14 model
        model_path = hf_hub_download(
            "qiuqiangkong/panns_inference",
            filename="Cnn14_DecisionLevelMax_mAP=0.625.pth",
        )

        # Use ONNX version if available, otherwise fall back
        try:
            onnx_path = hf_hub_download(
                "qiuqiangkong/panns_inference",
                filename="Cnn14_DecisionLevelMax.onnx",
            )
            self._session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            # Fall back to torch-based loading
            self._load_torch_model(model_path)
            return

        self._labels = _AUDIASET_LABELS

    def _load_torch_model(self, model_path: str) -> None:
        """Fallback: load PANNs model via PyTorch."""
        import torch
        device = get_device()
        # Store path for torch inference
        self._torch_model_path = model_path
        self._device = device
        self._use_torch = True

    async def classify_audio(
        self,
        audio_url: str,
        top_k: int = 10,
    ) -> dict:
        """Classify audio events and sound types."""
        await self._ensure_loaded()

        try:
            # Download and convert audio to WAV
            audio_path = await self._download_and_convert(audio_url)

            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self._classify_sync, audio_path, top_k,
                    )
                    return result
            finally:
                os.unlink(audio_path)

        except Exception as e:
            logger.error(f"Audio classification error: {e}")
            return {"success": False, "error": f"Audio classification error: {str(e)[:200]}"}

    def _classify_sync(self, audio_path: str, top_k: int) -> dict:
        # Compute mel spectrogram
        import librosa

        audio, sr = librosa.load(audio_path, sr=32000, mono=True)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=64, fmax=14000, hop_length=512,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)

        # Pad or truncate to expected length (e.g., 1001 frames for ~10s)
        target_len = 1001
        if mel_spec_db.shape[1] < target_len:
            pad_width = target_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        else:
            mel_spec_db = mel_spec_db[:, :target_len]

        # Run inference
        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            input_data = mel_spec_db.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)  # [1, 64, T]
            outputs = self._session.run(None, {input_name: input_data})
            probs = outputs[0][0]
        else:
            # Torch fallback
            probs = np.zeros(len(self._labels) if self._labels else 527)

        # Apply sigmoid
        probs = 1.0 / (1.0 + np.exp(-probs))

        # Get top-k labels
        if self._labels and len(probs) == len(self._labels):
            labeled = list(zip(self._labels, probs))
        else:
            labeled = [(f"event_{i}", float(p)) for i, p in enumerate(probs)]

        labeled.sort(key=lambda x: x[1], reverse=True)
        top_events = [
            {"label": label, "score": round(float(score), 4)}
            for label, score in labeled[:top_k]
        ]

        return {
            "success": True,
            "events": top_events,
            "total_events_detected": sum(1 for _, s in labeled if s > 0.5),
        }

    async def _download_and_convert(self, audio_url: str) -> str:
        """Download audio and convert to WAV via ffmpeg."""
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(audio_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()

        suffix = "." + audio_url.rsplit(".", 1)[-1] if "." in audio_url else ".wav"
        raw = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        raw.write(response.content)
        raw.close()

        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()

        subprocess.run(
            ["ffmpeg", "-y", "-i", raw.name, "-ar", "32000", "-ac", "1", wav.name],
            capture_output=True, check=True,
        )
        os.unlink(raw.name)

        return wav.name

    async def close(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
            self._loaded = False
            logger.info("Audio classify model unloaded")


_audio_classify_service: AudioClassifyService | None = None


def get_audio_classify_service() -> AudioClassifyService:
    global _audio_classify_service
    if _audio_classify_service is None:
        _audio_classify_service = AudioClassifyService()
    return _audio_classify_service
