"""ComfyUI Ray deployment — subprocess proxy within KubeRay worker pod.

Ray actor starts ComfyUI as a subprocess and proxies HTTP requests.
The ComfyUI image has /opt/ComfyUI pre-installed — we just launch it.
"""
from __future__ import annotations

import logging
import subprocess

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
        return await self._proxy_request(request)
