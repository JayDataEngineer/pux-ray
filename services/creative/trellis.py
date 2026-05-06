"""TRELLIS StableProjectorz — Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Runs as a subprocess within the KubeRay worker pod.
"""
from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, SubprocessProxyMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class TRELLISDeployment(BaseGPUDeployment, SubprocessProxyMixin):
    """TRELLIS image-to-3D via subprocess proxy."""

    SUBPROCESS_PORT = 9000

    def _load(self, model_name: str = "trellis-spz") -> None:
        import subprocess as sp

        # Apply runtime patches (same as start_trellis.sh)
        sp.run(["bash", "-c",
                "cd /opt/trellis && "
                "git checkout HEAD -- trellis2/pipelines/rembg/BiRefNet.py "
                "trellis2/modules/image_feature_extractor.py "
                "trellis2/modules/sparse/conv/conv_flex_gemm.py "
                "trellis2/representations/mesh/base.py 2>/dev/null || true"],
               check=False)
        for patch in ["/opt/patch_birefnet_runtime.py", "/opt/patch_dinov3_runtime.py",
                      "/opt/patch_conv_flex_gemm.py", "/opt/patch_cumesh_fallback.py"]:
            sp.run(["python3", patch], check=False)

        self._start_proxy(
            cmd=["python3", "api_spz/main_api.py",
                 "--host", "0.0.0.0", "--port", str(self.SUBPROCESS_PORT)],
            port=self.SUBPROCESS_PORT,
            health_path="/health",
            timeout=600,
            cwd="/opt/trellis",
            env={
                "TRELLIS_MODEL_ID": "/models/3d/trellis/TRELLIS.2-4B",
                "HF_HOME": "/root/.cache/huggingface",
                "ATTN_BACKEND": "xformers",
                "PYTHONPATH": "/app:/opt/trellis:/opt/utils3d",
            },
        )
        self.model = True
        self.model_name = model_name

    def _unload(self) -> None:
        self._stop_proxy()
        self.model = None

    async def __call__(self, request):
        await self._ensure_loaded("trellis-spz")
        return await self._proxy_request(request)
