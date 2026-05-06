"""Florence-2-large-ft - Vision foundation model.

0.77B parameter vision model. Captioning, object detection, segmentation,
OCR, region grounding.
Requires ~2GB VRAM.
"""
from __future__ import annotations

import io
import logging
import os

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, _free_cuda_cache

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
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class Florence2Deployment(BaseGPUDeployment):
    """Florence-2 vision model."""

    def _load(self, model_name: str = "florence2-large-ft") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Florence-2 model not found at {MODEL_PATH}")

        from transformers import AutoProcessor, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        self.model_name = model_name
        logger.info("Florence-2 loaded from %s on %s", MODEL_PATH, self.device)

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        _free_cuda_cache()

    async def __call__(self, request):
        if not self.is_loaded():
            self.load_model("florence2-large-ft")

        import base64
        from PIL import Image

        body = await request.json()
        task = body.get("task", "caption")
        task_prompt = TASK_PROMPTS.get(task, "<CAPTION>")

        image_data = body.get("image")
        if not image_data:
            return JSONResponse({"error": "image is required"}, status_code=400)

        text_input = body.get("text_input")

        if image_data.startswith("http"):
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(image_data)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
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

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height),
        )

        return JSONResponse(parsed_answer)
