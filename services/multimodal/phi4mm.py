"""Phi-4-multimodal-instruct - Omni model (text + vision + speech → text).

5.6B parameter multimodal model. Processes text, image, and audio inputs.
Runs inside Ray-managed container (tech-noir/phi4mm:latest).

Requires ~16GB VRAM.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

import torch
from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment, _free_cuda_cache, container_runtime

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("PHI4MM_MODEL_PATH", "/models/multimodal/phi4-multimodal-instruct")


@serve.deployment(
    name="phi4mm",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": container_runtime("tech-noir/phi4mm:latest"),
    },
)
class Phi4MMDeployment(BaseGPUDeployment):
    """Phi-4-multimodal via Ray native container. Text + image + audio → text."""

    def _load(self, model_name: str = "phi4-multimodal") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Phi-4-multimodal model not found at {MODEL_PATH}")

        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        self.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model_name = model_name
        logger.info("Phi-4-multimodal loaded from %s", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        self.generation_config = None
        _free_cuda_cache()

    async def __call__(self, request):
        body = await request.json()
        mode = body.get("mode", "text")

        try:
            if mode == "text":
                return await self._handle_text(body)
            elif mode == "vision":
                return await self._handle_vision(body)
            elif mode == "audio":
                return await self._handle_audio(body)
            elif mode == "vision_audio":
                return await self._handle_vision_audio(body)
            else:
                return JSONResponse({"error": f"Unknown mode: {mode}"}, status_code=400)
        except Exception as e:
            logger.exception("Phi-4mm inference failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def _handle_text(self, body: dict) -> Response:
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1000)
        temperature = body.get("temperature", 0.7)

        from transformers import GenerationConfig

        prompt = self._format_chat(messages)
        inputs = self.processor(text=prompt, return_tensors="pt").to("cuda")

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        generate_ids = self.model.generate(**inputs, generation_config=gen_config)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return JSONResponse({"content": response_text})

    async def _handle_vision(self, body: dict) -> Response:
        import base64
        from PIL import Image

        prompt_text = body.get("prompt", "Describe the image in detail.")
        image_data = body.get("image")
        max_tokens = body.get("max_tokens", 1000)

        if not image_data:
            return JSONResponse({"error": "image is required for vision mode"}, status_code=400)

        if image_data.startswith("http"):
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(image_data)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"<|user|><|image_1|>{prompt_text}<|end|><|assistant|>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return JSONResponse({"content": response_text})

    async def _handle_audio(self, body: dict) -> Response:
        prompt_text = body.get("prompt", "Transcribe the audio clip into text.")
        audio_data = body.get("audio")
        audio_sr = body.get("audio_sample_rate")
        max_tokens = body.get("max_tokens", 1000)

        if not audio_data:
            return JSONResponse({"error": "audio is required for audio mode"}, status_code=400)

        import base64
        import tempfile
        import soundfile as sf

        raw_audio = base64.b64decode(audio_data)

        if audio_sr is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_audio)
                tmp_path = tmp.name
            audio, sr = sf.read(tmp_path)
            os.unlink(tmp_path)
        else:
            import io as _io
            audio, sr = sf.read(_io.BytesIO(raw_audio))

        prompt = f"<|user|><|audio_1|>{prompt_text}<|end|><|assistant|>"
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors="pt").to("cuda")

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return JSONResponse({"content": response_text})

    async def _handle_vision_audio(self, body: dict) -> Response:
        import base64
        from PIL import Image
        import tempfile
        import soundfile as sf

        image_data = body.get("image")
        audio_data = body.get("audio")
        audio_sr = body.get("audio_sample_rate")
        prompt_text = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 1000)

        if not image_data or not audio_data:
            return JSONResponse({"error": "both image and audio are required"}, status_code=400)

        if image_data.startswith("http"):
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(image_data)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        raw_audio = base64.b64decode(audio_data)
        if audio_sr is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_audio)
                tmp_path = tmp.name
            audio, sr = sf.read(tmp_path)
            os.unlink(tmp_path)
        else:
            import io as _io
            audio, sr = sf.read(_io.BytesIO(raw_audio))

        prompt = f"<|user|><|image_1|><|audio_1|>{prompt_text}<|end|><|assistant|>"
        inputs = self.processor(text=prompt, images=image, audios=[(audio, sr)], return_tensors="pt").to("cuda")

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return JSONResponse({"content": response_text})

    @staticmethod
    def _format_chat(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}<|end|>")
        parts.append("<|assistant|>")
        return "".join(parts)
