"""AniGen — Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Runs as a subprocess within the KubeRay worker pod.
"""
from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class AniGenDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """AniGen animated 3D asset generation via subprocess proxy."""

    SUBPROCESS_PORT = 9000

    def _load(self, model_name: str = "anigen") -> None:
        import subprocess as sp

        # Link models (same as entrypoint_anigen.sh)
        if not sp.run(["bash", "-c",
                "[ -d /models/ckpts/ckpts ] && [ ! -e /opt/anigen/ckpts ] && "
                "ln -s /models/ckpts/ckpts /opt/anigen/ckpts || true"],
               check=False).returncode:
            pass

        self._start_proxy(
            cmd=["python", "-m", "uvicorn", "api:app",
                 "--host", "0.0.0.0", "--port", str(self.SUBPROCESS_PORT)],
            port=self.SUBPROCESS_PORT,
            health_path="/health",
            timeout=600,
            cwd="/opt",
            env={
                "ANIGEN_MODEL_DIR": "/models/3d/anigen/ckpts",
                "HF_HOME": "/root/.cache/huggingface",
                "PYTHONPATH": "/app:/opt/anigen:/opt/utils3d-src:/opt",
            },
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("anigen")
        return await self._proxy_request(request)
