"""GPT-SoVITS — Voice cloning TTS.

Clones voices from reference audio using GPT-SoVITS.
Runs as a subprocess within the KubeRay worker pod.
"""
from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class GPTSoVITSDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """GPT-SoVITS voice cloning TTS via subprocess proxy."""

    SUBPROCESS_PORT = 9000

    def _load(self, model_name: str = "gpt-sovits") -> None:
        self._start_proxy(
            cmd=["python", "-m", "uvicorn", "api:app",
                 "--host", "0.0.0.0", "--port", str(self.SUBPROCESS_PORT)],
            port=self.SUBPROCESS_PORT,
            health_path="/health",
            timeout=300,
            cwd="/opt",
            env={"PYTHONPATH": "/app:/opt/gpt-sovits:/opt"},
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("gpt-sovits")
        return await self._proxy_request(request)
