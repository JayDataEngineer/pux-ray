"""ComfyUI Forge service — subprocess proxy with workflow adapter.

Used by the Forge (services/forge.py SERVICE_MAP) for ComfyUI image generation.
Runs ComfyUI as a subprocess and proxies requests.

Supports two modes:
1. Workflow submission — send {"workflow": <comfyui_api_format>} to queue
   a workflow, wait for completion, and return output images as base64.
2. Raw API proxy — send {"path": "/api/..."} to call ComfyUI's HTTP API.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import uuid
from typing import Any

from services.forge_base import ForgeService
from services.forge_subprocess import ForgeSubprocessMixin

logger = logging.getLogger(__name__)


class ComfyUIService(ForgeSubprocessMixin, ForgeService):
    """Forge-managed ComfyUI service via subprocess proxy."""
    vram_mb = 14_336
    service_name = "comfyui"
    default_model = "comfyui"

    PORT = 18465

    def load(self, model_name: str, quant: str | None = None) -> None:
        # Symlink model dirs from mounted /models volume.
        # Directories that only contain pre-downloaded files are symlinked as-is.
        # Directories where custom nodes may auto-download files at runtime
        # (RMBG, controlnet, sams, ultralytics) keep writable local dirs with
        # their pre-downloaded content symlinked inside. This avoids read-only
        # filesystem errors when nodes try to write model cache/config files.
        subprocess.run(["bash", "-c", """
            set -e
            MODELS="/opt/ComfyUI/models"
            SOURCE="/models/image-gen/comfyui"
            mkdir -p "$MODELS"

            # Fully symlink — read-only, pre-downloaded only
            for d in checkpoints clip clip_vision diffusion_models loras \
                     text_encoders unet upscale_models vae latent_upscale_models \
                     HY-Motion; do
              target="$SOURCE/$d"
              link="$MODELS/$d"
              if [ -d "$target" ]; then
                [ -L "$link" ] || [ -d "$link" ] && rm -rf "$link"
                ln -s "$target" "$link"
              fi
            done

            # Writable dirs — node may auto-download files at runtime.
            # Subdirectories are NOT symlinked — each is created as a real
            # writable dir with individual model files symlinked inside.
            # This allows custom nodes to write cache/config files (.cache,
            # config.json, etc.) inside the subdirectory.
            for d in controlnet RMBG sams ultralytics; do
              target="$SOURCE/$d"
              link="$MODELS/$d"
              mkdir -p "$link"
              if [ -d "$target" ]; then
                # Walk all files recursively, recreate dir structure as real dirs
                find "$target" -type f \( -name '*.safetensors' -o -name '*.pth' \
                  -o -name '*.pt' -o -name '*.bin' -o -name '*.onnx' \) | while read -r f; do
                  rel="${f#$target/}"
                  parent="$(dirname "$rel")"
                  destdir="$link/$parent"
                  mkdir -p "$destdir"
                  flink="$destdir/$(basename "$f")"
                  [ -L "$flink" ] || [ -f "$flink" ] && rm -f "$flink"
                  ln -s "$f" "$flink" 2>/dev/null || true
                done
              fi
            done
        """], check=False)

        self.start_subprocess(
            cmd=["python3", "main.py", "--port", str(self.PORT),
                 "--listen", "0.0.0.0", "--preview-method", "auto",
                 "--use-sage-attention", "--enable-manager"],
            port=self.PORT,
            health_path="/",
            timeout=900,
            cwd="/opt/ComfyUI",
        )
        self.model_name = model_name
        self._loaded = True

    def unload(self) -> None:
        self.stop_subprocess()
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        if "workflow" in payload:
            return self._handle_workflow_sync(payload)

        if "path" in payload:
            try:
                if payload.get("raw"):
                    kwargs: dict[str, Any] = {
                        "method": payload.get("method", "GET"),
                        "path": payload["path"],
                        "params": payload.get("params"),
                    }
                    body = payload.get("body")
                    if body is not None:
                        if isinstance(body, dict):
                            kwargs["json"] = body
                        else:
                            kwargs["content"] = body
                    return self._call_raw_full(**kwargs)
                result = self._call(
                    method=payload.get("method", "GET"),
                    path=payload["path"],
                    json=payload.get("body"),
                    params=payload.get("params"),
                )
                return {"status": "ok", "result": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        return {"status": "error", "error": "payload must contain 'workflow' or 'path'"}

    def _handle_workflow_sync(self, payload: dict) -> dict:
        """Synchronous workflow submission — queue, poll, return outputs."""
        import time as _time

        workflow = payload["workflow"]
        client_id = payload.get("client_id", str(uuid.uuid4()))
        timeout_sec = payload.get("timeout", 600)
        poll_interval = payload.get("poll_interval", 1.0)

        try:
            resp = self._call("POST", "/prompt", timeout=30,
                              json={"prompt": workflow, "client_id": client_id})
        except Exception as e:
            return {"status": "error", "error": f"Failed to queue workflow: {e}"}

        prompt_id = resp.get("prompt_id")
        logger.info("ComfyUI workflow queued: prompt_id=%s", prompt_id)

        deadline = _time.time() + timeout_sec
        while _time.time() < deadline:
            _time.sleep(poll_interval)
            try:
                hist_data = self._call("GET", f"/history/{prompt_id}", timeout=10)
            except Exception:
                continue

            if prompt_id not in hist_data:
                continue

            entry = hist_data[prompt_id]
            status = entry.get("status", {})
            status_str = status.get("status_str", "")

            if status_str == "success":
                outputs = entry.get("outputs", {})
                images = []
                for node_id, node_out in outputs.items():
                    for img in node_out.get("images", []):
                        try:
                            img_bytes = self._call_raw("GET", "/view", timeout=30, params={
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output"),
                            })
                            images.append({
                                "node_id": node_id,
                                "filename": img["filename"],
                                "content_type": "image/png",
                                "data": base64.b64encode(img_bytes).decode(),
                            })
                        except Exception:
                            pass

                logger.info("ComfyUI workflow done: prompt_id=%s images=%d", prompt_id, len(images))
                return {
                    "status": "ok",
                    "prompt_id": prompt_id,
                    "outputs": {
                        k: {kk: vv for kk, vv in v.items() if kk != "images"}
                        for k, v in outputs.items()
                    },
                    "images": images,
                }

            if status_str == "error":
                return {"status": "error", "error": "Workflow execution failed",
                        "prompt_id": prompt_id, "details": entry}

        return {"status": "error", "error": "Workflow timed out",
                "prompt_id": prompt_id, "timeout": timeout_sec}
