"""Media Analysis MCP Server — Ray Serve subprocess wrapper.

Runs the media-analysis-mcp FastMCP server as a managed subprocess.
Provides vision, audio, and video analysis tools via HTTP/SSE.

Accessible at:
  HTTP:  POST /mcp/media/*  (proxied through ingress)
  MCP:   http://localhost:8101/mcp
"""

from __future__ import annotations

import logging
from pathlib import Path

from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from services.base import SubprocessMixin
from registry.config import Config

logger = logging.getLogger(__name__)

MEDIA_MCP_PORT = 8101


@serve.deployment(
    name="media_analysis_mcp",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    max_ongoing_requests=8,
)
class MediaAnalysisMCPDeployment(SubprocessMixin):
    """Manages the media-analysis-mcp subprocess."""

    def __init__(self):
        super().__init__()
        self._running = False
        self.port = Config().get("services.mcp.media.port", MEDIA_MCP_PORT)
        self._venv_python = Config().get(
            "services.mcp.media.venv_python",
            "infra/repos/media-analysis-mcp/.venv/bin/python",
        )
        self._working_dir = Config().get(
            "services.mcp.media.working_dir",
            "infra/repos/media-analysis-mcp",
        )

    def is_running(self) -> bool:
        return self._running and self.process is not None and self.process.poll() is None

    def start_server(self) -> bool:
        """Start the MCP server subprocess."""
        if self.is_running():
            return True

        working_dir = Path(self._working_dir)
        if not working_dir.is_absolute():
            working_dir = Path(Config().project_root) / working_dir

        cmd = [
            self._venv_python, "-m", "uvicorn",
            "src.server:app",
            "--host", "0.0.0.0",
            "--port", str(self.port),
        ]

        self.start_process(cmd, cwd=str(working_dir))
        self._running = True
        logger.info("Media Analysis MCP server started on port %d", self.port)
        return True

    def stop_server(self) -> None:
        """Stop the MCP server subprocess."""
        self.stop_process()
        self._running = False
        logger.info("Media Analysis MCP server stopped")

    async def __call__(self, request: Request) -> Response:
        """Proxy all requests to the MCP server."""
        if not self._running:
            self.start_server()
            self.wait_for_health(
                f"http://127.0.0.1:{self.port}/health",
                timeout=120,
            )
        return await self._proxy_request(request)

    async def _proxy_request(self, request: Request) -> Response:
        """Forward request to the MCP server."""
        import httpx

        url = f"http://127.0.0.1:{self.port}{request.url.path}"
        # Strip the Ray Serve route prefix
        url = url.replace("/mcp/media", "")

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=await request.body(),
                params=dict(request.query_params),
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )


app = MediaAnalysisMCPDeployment.bind()
