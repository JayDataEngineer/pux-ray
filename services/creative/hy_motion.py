"""HY-Motion 1.0 — Text-to-3D human motion generation.

Generates skeleton-based 3D character animations from text prompts.
Runs as a subprocess within the KubeRay worker pod.
"""
from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="hy_motion",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class HYMotionDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """HY-Motion text-to-3D motion via subprocess proxy."""

    SUBPROCESS_PORT = 9000

    def _load(self, model_name: str = "hy-motion-1.0") -> None:
        self._start_proxy(
            cmd=["python", "-m", "uvicorn", "api:app",
                 "--host", "0.0.0.0", "--port", str(self.SUBPROCESS_PORT)],
            port=self.SUBPROCESS_PORT,
            health_path="/health",
            timeout=600,
            cwd="/opt",
            env={
                "HYMOTION_MODEL_PATH": "/models/image-gen/comfyui/HY-Motion/ckpts/tencent/HY-Motion-1.0-Lite",
                "PYTHONPATH": "/app:/opt/hymotion:/opt",
                "USE_HF_MODELS": "1",
            },
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("hy-motion-1.0")
        return await self._proxy_request(request)
