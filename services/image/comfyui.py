"""ComfyUI Ray deployment — subprocess proxy with workflow adapter.

Ray actor starts ComfyUI as a subprocess and proxies HTTP requests.
The ComfyUI image has /opt/ComfyUI pre-installed — we just launch it.

Supports two modes:
1. Workflow submission — send {"workflow": <comfyui_api_format>} to queue
   a workflow, wait for completion, and return output images as base64.
2. Raw API proxy — any other request is proxied directly to ComfyUI's
   HTTP API (/prompt, /queue, /history, /view, /system_stats, etc.).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import uuid

import httpx
from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="comfyui",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class ComfyUIDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """ComfyUI via subprocess proxy within KubeRay worker pod."""
    vram_mb = 14_336
    _service_name = "comfyui"

    PORT = 18465

    def _load(self, model_name: str = "comfyui") -> None:
        # Symlink model dirs from mounted /models volume
        subprocess.run(["bash", "-c", """
            mkdir -p /opt/ComfyUI/models
            for d in HY-Motion RMBG sams ultralytics; do
              [ -d /models/image-gen/comfyui/$d ] && [ ! -e /opt/ComfyUI/models/$d ] &&
                ln -s /models/image-gen/comfyui/$d /opt/ComfyUI/models/$d
            done
        """], check=False)

        self._start_proxy(
            cmd=["python3", "main.py", "--port", str(self.PORT),
                 "--listen", "0.0.0.0", "--preview-method", "auto",
                 "--use-split-cross-attention"],
            port=self.PORT,
            health_path="/",
            timeout=900,
            cwd="/opt/ComfyUI",
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("comfyui")

        try:
            body = await request.json()
        except Exception:
            return await self._proxy_request(request)

        # Mode 1: Workflow submission — queue workflow, wait, return outputs
        if "workflow" in body:
            return await self._handle_workflow(body)

        # Mode 2: Raw API proxy — forward to ComfyUI's native HTTP API
        # The caller specifies the target path explicitly.
        if "path" in body:
            return await self._api_call(
                method=body.get("method", "GET"),
                path=body["path"],
                json_body=body.get("body"),
                params=body.get("params"),
            )

        # Mode 3: Default proxy — forward the request as-is
        return await self._proxy_request(request)

    async def _api_call(
        self,
        method: str = "GET",
        path: str = "/system_stats",
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> "Response":
        from starlette.responses import JSONResponse, Response

        if not path.startswith("/"):
            path = f"/{path}"

        target = f"{self._proxy_base_url}{path}"
        timeout = httpx.Timeout(600.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=method.upper(),
                url=target,
                json=json_body,
                params=params,
            )

        from starlette.responses import Response as StarletteResponse
        return StarletteResponse(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _handle_workflow(self, payload: dict) -> "Response":
        from starlette.responses import JSONResponse

        workflow = payload["workflow"]
        client_id = payload.get("client_id", str(uuid.uuid4()))
        timeout_sec = payload.get("timeout", 600)
        poll_interval = payload.get("poll_interval", 1.0)

        timeout_cfg = httpx.Timeout(float(timeout_sec), connect=10.0)

        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            # Queue the workflow
            resp = await client.post(
                f"{self._proxy_base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if resp.status_code != 200:
                return JSONResponse(
                    {"status": "error", "error": "Failed to queue workflow",
                     "details": resp.text},
                    status_code=resp.status_code,
                )

            prompt_id = resp.json().get("prompt_id")
            logger.info("ComfyUI workflow queued: prompt_id=%s", prompt_id)

            # Poll /history/{prompt_id} until completion
            max_polls = int(timeout_sec / poll_interval)
            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                hist_resp = await client.get(
                    f"{self._proxy_base_url}/history/{prompt_id}",
                )
                if hist_resp.status_code != 200:
                    continue

                hist_data = hist_resp.json()
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
                            img_resp = await client.get(
                                f"{self._proxy_base_url}/view",
                                params={
                                    "filename": img["filename"],
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output"),
                                },
                            )
                            if img_resp.status_code == 200:
                                images.append({
                                    "node_id": node_id,
                                    "filename": img["filename"],
                                    "content_type": img_resp.headers.get(
                                        "content-type", "image/png"
                                    ),
                                    "data": base64.b64encode(
                                        img_resp.content
                                    ).decode(),
                                })

                    logger.info(
                        "ComfyUI workflow done: prompt_id=%s images=%d",
                        prompt_id, len(images),
                    )
                    return JSONResponse({
                        "status": "ok",
                        "prompt_id": prompt_id,
                        "outputs": {
                            k: {kk: vv for kk, vv in v.items() if kk != "images"}
                            for k, v in outputs.items()
                        },
                        "images": images,
                    })

                if status_str == "error":
                    return JSONResponse(
                        {"status": "error", "error": "Workflow execution failed",
                         "prompt_id": prompt_id, "details": entry},
                        status_code=500,
                    )

            return JSONResponse(
                {"status": "error", "error": "Workflow timed out",
                 "prompt_id": prompt_id, "timeout": timeout_sec},
                status_code=504,
            )


comfyui = ComfyUIDeployment.bind()
