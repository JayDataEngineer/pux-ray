"""ShowUI-2B service — GUI grounding: screenshot → click coordinates.

Uses showlab/ShowUI-2B (Qwen2-VL-2B fine-tuned on UI grounding) to locate
UI elements described in natural language. Returns normalized [x, y] coordinates
plus absolute pixel coordinates for the image dimensions.

Always runs on CPU (forced, regardless of MEDIA_DEVICE) to preserve VRAM
for GPU services on the shared RTX 4090.

Also supports ByteDance-Seed/UI-TARS-1.5-7B as an alternative via
MEDIA_SHOWUI_MODEL env var (requires transformers >= 4.49.0 for Qwen2.5-VL).
"""

import asyncio
import json
import re
import time
from typing import Optional

from loguru import logger

from ..settings import get_settings
from .idle_watcher import get_idle_watcher


_SYSTEM_PROMPT = (
    "Based on the screenshot of the page, I give a text description and you give its "
    "corresponding location. The coordinate represents a clickable location [x, y] for "
    "an element, which is a relative coordinate on the screenshot, scaled to a range of 0 to 1."
)


def _parse_coordinates(raw: str) -> tuple[float, float] | None:
    """Parse model output to (x_norm, y_norm) in 0–1 range.

    Handles two output formats:
    - ShowUI-2B:     [0.532, 0.234]
    - UI-TARS-1.5:  <|box_start|>(532,234)<|box_end|>  (0–1000 scale)
    """
    text = raw.strip()

    # UI-TARS format: <|box_start|>(x,y)<|box_end|>
    m = re.search(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>", text)
    if m:
        return int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0

    # JSON list: [0.532, 0.234]  or  [532, 234] (0-1000 scale when > 1)
    try:
        coords = json.loads(text)
        if isinstance(coords, list) and len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            if x > 1.0 or y > 1.0:
                x, y = x / 1000.0, y / 1000.0
            return x, y
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback regex: two numbers separated by comma
    m = re.search(r"\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?", text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x > 1.0 or y > 1.0:
            x, y = x / 1000.0, y / 1000.0
        return x, y

    return None


def _load_qwen2vl(model_name: str, min_pixels: int, max_pixels: int):
    """Load Qwen2-VL model + processor (transformers 4.45+, covers ShowUI-2B)."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cpu")
    model.eval()
    return model, processor


def _load_qwen25vl(model_name: str, min_pixels: int, max_pixels: int):
    """Load Qwen2.5-VL model + processor (transformers 4.49+, covers UI-TARS-1.5)."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cpu")
    model.eval()
    return model, processor


class ShowUIService:

    def __init__(self):
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("showui"):
                raise RuntimeError("ShowUI is disabled")

            try:
                logger.info(f"Loading ShowUI: {settings.showui_model} (CPU-only, saves VRAM)")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)
                elapsed = time.time() - start
                logger.info(f"ShowUI loaded in {elapsed:.1f}s")
                self._loaded = True
                get_idle_watcher().watch("showui", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load ShowUI: {e}")
                raise

    def _load_model_sync(self) -> None:
        settings = get_settings()
        model_name = settings.showui_model
        min_px = settings.showui_min_pixels
        max_px = settings.showui_max_pixels

        # UI-TARS-1.5 is Qwen2.5-VL; ShowUI-2B is Qwen2-VL
        is_uitars = "UI-TARS-1.5" in model_name or "ui-tars-1.5" in model_name.lower()

        if is_uitars:
            logger.info("Detected UI-TARS-1.5 — using Qwen2.5-VL loader (needs transformers >= 4.49)")
            self._model, self._processor = _load_qwen25vl(model_name, min_px, max_px)
        else:
            self._model, self._processor = _load_qwen2vl(model_name, min_px, max_px)

    async def ground(
        self,
        image_url: str,
        query: str,
        max_new_tokens: int | None = None,
    ) -> dict:
        """Locate a UI element in a screenshot and return click coordinates.

        Returns:
            success, x, y (pixels), x_norm, y_norm (0-1), width, height, raw (model output)
        """
        try:
            await self._ensure_loaded()
        except RuntimeError as e:
            return {"success": False, "error": str(e)}

        get_idle_watcher().touch("showui")

        try:
            from .media_utils import load_image
            image = await load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        width, height = image.size
        settings = get_settings()
        n_tokens = max_new_tokens or settings.showui_max_new_tokens

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                inner = await asyncio.wait_for(
                    loop.run_in_executor(None, self._infer_sync, image, query, n_tokens),
                    timeout=300.0,
                )
            except asyncio.TimeoutError:
                return {"success": False, "error": "Inference timed out after 300s"}
            except Exception as e:
                logger.error(f"ShowUI inference error: {e}")
                return {"success": False, "error": f"Inference error: {str(e)[:200]}"}

        if "error" in inner:
            return {"success": False, "width": width, "height": height, **inner}

        x_norm, y_norm = inner["x_norm"], inner["y_norm"]
        return {
            "success": True,
            "width": width,
            "height": height,
            "x": round(x_norm * width),
            "y": round(y_norm * height),
            "x_norm": x_norm,
            "y_norm": y_norm,
            "raw": inner["raw"],
        }

    def _infer_sync(self, image, query: str, max_new_tokens: int) -> dict:
        import torch

        # Text messages only (image passed separately to processor)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # placeholder token; actual pixels via images=
                    {"type": "text", "text": query},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Convert to RGB — Qwen2-VL requires 3-channel input
        rgb = image.convert("RGB")

        inputs = self._processor(
            text=[text],
            images=[rgb],
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        raw = self._processor.decode(output_ids, skip_special_tokens=True).strip()

        coords = _parse_coordinates(raw)
        if coords is None:
            return {"raw": raw, "error": f"Could not parse coordinates from: {raw!r}"}

        x_norm, y_norm = coords
        return {
            "raw": raw,
            "x_norm": round(x_norm, 4),
            "y_norm": round(y_norm, 4),
        }

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            logger.info("ShowUI unloaded")


_showui_service: ShowUIService | None = None


def get_showui_service() -> ShowUIService:
    global _showui_service
    if _showui_service is None:
        _showui_service = ShowUIService()
    return _showui_service
