"""Phi-4-multimodal-instruct - Omni model (text + vision + speech -> text).

5.6B parameter multimodal model. Processes text, image, and audio inputs.
Requires ~16GB VRAM.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time

import torch
from PIL import Image
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("PHI4MM_MODEL_PATH", "/models/multimodal/phi4-multimodal-instruct")


@serve.deployment(
    name="phi4mm",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
            },
        },
    },
)
class Phi4MMDeployment(BaseGPUDeployment):
    """Phi-4-multimodal. Text + image + audio -> text."""

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
            _attn_implementation="sdpa",
            local_files_only=True,
        )
        self.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model_name = model_name
        logger.info("Phi-4-multimodal loaded from %s", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        self.generation_config = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, image_b64, audio_b64, mode}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "phi4-multimodal")

            mode = extracted.get("mode", "text")
            prompt_text = extracted.get("text", "")
            image_bytes = extracted.get("image")
            audio_bytes = extracted.get("audio")
            max_tokens = extracted.get("max_tokens", 1000)

            content = ""
            if mode == "text":
                messages = extracted.get("messages", [])
                prompt = self._format_chat(messages)
                content = await asyncio.to_thread(
                    self._run_text_inference, prompt, max_tokens, 0.7,
                )
            elif mode == "vision":
                if not image_bytes:
                    return JSONResponse(self.handle_error("image_b64 required for vision mode"), status_code=400)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                content = await asyncio.to_thread(
                    self._run_vision_inference, image, prompt_text or "Describe the image in detail.", max_tokens,
                )
            elif mode == "audio":
                if not audio_bytes:
                    return JSONResponse(self.handle_error("audio_b64 required for audio mode"), status_code=400)
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                content = await asyncio.to_thread(
                    self._run_audio_inference, audio, sr, prompt_text or "Transcribe the audio clip into text.", max_tokens,
                )
            elif mode == "vision_audio":
                if not image_bytes or not audio_bytes:
                    return JSONResponse(self.handle_error("image_b64 and audio_b64 both required"), status_code=400)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                content = await asyncio.to_thread(
                    self._run_vision_audio_inference, image, audio, sr, prompt_text, max_tokens,
                )
            else:
                return JSONResponse(self.handle_error(f"Unknown mode: {mode}"), status_code=400)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    json.dumps({"content": content, "mode": mode}).encode("utf-8"),
                    "application/json",
                    latency_ms,
                    extra_metrics={"mode": mode},
                )
            )
        except Exception as e:
            logger.error("phi4mm error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    def _run_text_inference(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from transformers import GenerationConfig

        inputs = self.processor(text=prompt, return_tensors="pt").to("cuda")
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        generate_ids = self.model.generate(**inputs, generation_config=gen_config)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def _run_vision_inference(self, image: "Image.Image", prompt_text: str, max_tokens: int) -> str:
        prompt = f"<|image_1|>{prompt_text}<|end|><|assistant|"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_tokens, generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def _run_audio_inference(self, audio, sr: int, prompt_text: str, max_tokens: int) -> str:
        prompt = f"<|audio_1|>{prompt_text}<|end|><|assistant|"
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_tokens, generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def _run_vision_audio_inference(self, image: "Image.Image", audio, sr: int, prompt_text: str, max_tokens: int) -> str:
        prompt = f"<|image_1|><|audio_1|>{prompt_text}<|end|><|assistant|"
        inputs = self.processor(text=prompt, images=image, audios=[(audio, sr)], return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_tokens, generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    @staticmethod
    def _format_chat(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}<|end|>")
        parts.append("<|assistant|")
        return "".join(parts)