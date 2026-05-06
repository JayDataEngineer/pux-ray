"""See-Through — Layer decomposition for anime character illustrations.

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
Runs as a subprocess within the KubeRay worker pod.
Requires ~4GB VRAM.
"""
from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class SeeThroughDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """See-Through layer decomposition via subprocess proxy."""

    SUBPROCESS_PORT = 9000

    def _load(self, model_name: str = "see-through") -> None:
        self._start_proxy(
            cmd=["python", "-m", "uvicorn", "api:app",
                 "--host", "0.0.0.0", "--port", str(self.SUBPROCESS_PORT)],
            port=self.SUBPROCESS_PORT,
            health_path="/health",
            timeout=300,
            cwd="/opt",
            env={
                "SEETHROUGH_MODEL_DIR": "/models/vision/see-through",
                "HF_HOME": "/root/.cache/huggingface",
                "PYTHONPATH": "/app:/opt/seethrough:/opt",
            },
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("see-through")
        return await self._proxy_request(request)
