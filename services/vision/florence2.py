"""Florence-2-large-ft - Vision foundation model.

0.77B parameter vision model. Captioning, object detection, segmentation,
OCR, region grounding.
Requires ~2GB VRAM.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time

import torch
from PIL import Image
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("FLORENCE2_MODEL_PATH", "/models/vision/florence-2-large-ft")

TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
}


@serve.deployment(
    name="florence2",
    num_replicas=1,
    max_ongoing_requests=2,
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
class Florence2Deployment(BaseGPUDeployment):
    """Florence-2 vision model."""

    def _load(self, model_name: str = "florence2-large-ft") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Florence-2 model not found at {MODEL_PATH}")

        from transformers import AutoProcessor, AutoModelForCausalLM

        # Patch _supports_sdpa on all model classes to avoid attr errors
        # with newer transformers that removed this attribute
        import transformers.modeling_utils as _mu
        _mu.PreTrainedModel._supports_sdpa = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        ).to(self.device)

        # Patch: model code accesses past_key_values[0][0].shape without None check
        _orig_prep = type(self.model).prepare_inputs_for_generation
        def _safe_prep(self_inner, *args, past_key_values=None, **kwargs):
            kwargs["past_key_values"] = past_key_values
            # If past_key_values is None, inject empty tuple to avoid AttributeError
            if past_key_values is None:
                kwargs["past_key_values"] = ()
            return _orig_prep(self_inner, *args, **kwargs)
        type(self.model).prepare_inputs_for_generation = _safe_prep
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        self.model_name = model_name
        logger.info("Florence-2 loaded from %s on %s", MODEL_PATH, self.device)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        super()._unload()

    def _run_inference(self, image: "Image.Image", task_prompt: str, text_input: str | None) -> dict:
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype,
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]

        return self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height),
        )

    def _extract_input(self, inp) -> dict:
        result = super()._extract_input(inp)
        if inp.image_b64:
            from services.base import _b64_decode
            result["image"] = _b64_decode(inp.image_b64)
        return result

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {image_b64, task, text}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "florence2-large-ft")

            image_bytes = extracted.get("image")
            if not image_bytes:
                return JSONResponse(self.handle_error("image_b64 required"), status_code=400)

            task = extracted.get("task", "caption")
            task_prompt = TASK_PROMPTS.get(task, "<CAPTION>")
            text_input = extracted.get("text")

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if image.width < 64 or image.height < 64:
                image = image.resize((64, 64), Image.BILINEAR)

            parsed_answer = await asyncio.to_thread(
                self._run_inference, image, task_prompt, text_input
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    str(parsed_answer).encode("utf-8"),
                    "application/json",
                    latency_ms,
                    extra_metrics={"task": task},
                )
            )
        except Exception as e:
            logger.error("florence2 error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)