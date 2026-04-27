"""Deploy a lightweight ComfyUI proxy on Ray Serve for testing.

This connects to an already-running ComfyUI instance instead of managing
a subprocess. Used for testing that the Ray proxy pattern works before
committing to the full subprocess-managed deployment.

Usage:
    cd ~/Documents/programs/ray && .venv/bin/python -m services.image.comfyui_proxy_test
"""

from __future__ import annotations

import httpx
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

COMFYUI_URL = "http://127.0.0.1:8465"


@serve.deployment(
    name="comfyui",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class ComfyUIProxy:
    """Proxy requests to already-running ComfyUI."""

    def __init__(self):
        self.base_url = COMFYUI_URL
        print(f"ComfyUIProxy: proxying to {self.base_url}")

    async def __call__(self, request: Request) -> Response:
        async with httpx.AsyncClient(timeout=300) as client:
            path = request.url.path
            if path.startswith("/comfyui"):
                path = path[len("/comfyui"):] or "/"

            target_url = f"{self.base_url}{path}"
            if request.url.query:
                target_url += f"?{request.url.query}"

            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=body,
            )

            content_type = resp.headers.get("content-type", "application/json")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=content_type,
            )


if __name__ == "__main__":
    import ray
    ray.init(address="auto", ignore_reinit_error=True)

    from ray import serve
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    serve.run(ComfyUIProxy.bind(), name="comfyui", route_prefix="/comfyui")

    print("ComfyUI proxy deployed on http://localhost:8000/comfyui/*")
    print("Press Ctrl+C to stop")
    import signal
    signal.pause()
