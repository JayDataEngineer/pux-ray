"""Playground Ray Serve deployment — serves the interactive UI page + API."""

from __future__ import annotations

from pathlib import Path

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from gateway.dashboard import query_service_status
from gateway.playground import PLAYGROUND_META
from services.registry import get_service

_PLAYGROUND_HTML_PATH = Path(__file__).resolve().parent / "playground.html"


@serve.deployment(
    name="playground",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1},
)
class PlaygroundDeployment:

    async def __call__(self, request: Request) -> Response:
        path = request.url.path.rstrip("/")

        if path.endswith("/api/services"):
            return await self._services()

        return HTMLResponse(_PLAYGROUND_HTML_PATH.read_text())

    async def _services(self) -> JSONResponse:
        deploy_status = {s["name"]: s for s in query_service_status()}
        services = []
        for name, meta in PLAYGROUND_META.items():
            entry = get_service(name)
            dep = deploy_status.get(entry.deployment if entry else name, {})
            services.append({
                "name": name,
                "label": entry.label if entry else name,
                "category": meta["category"],
                "gpu": meta["gpu"],
                "route": meta.get("route"),
                "format": meta.get("format", "tnap"),
                "input_fields": meta["input_fields"],
                "output_type": entry.output_type if entry else "unknown",
                "description": entry.description if entry else "",
                "status": dep.get("status", "UNKNOWN"),
                "running_replicas": dep.get("running_replicas", 0),
            })
        return JSONResponse(services)
