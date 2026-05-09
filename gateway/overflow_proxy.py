"""Overflow proxy — bridges local Ray Serve and SkyServe cloud endpoint.

Deployed as a standalone k8s Deployment in the infra namespace.
Traefik routes overflow requests here when the local cluster is overloaded.

Flow:
  1. Receive request
  2. Try local Ray Serve
  3. On 503/timeout → forward to SkyServe cloud endpoint
  4. Return whichever responded first (with Langfuse tracing tag)
"""
from __future__ import annotations

import logging
import os
import time

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

LOCAL_RAY_URL = os.environ.get("LOCAL_RAY_URL", "http://tech-noir-ray-serve-svc.ai-services.svc.cluster.local:8000")
CLOUD_SERVE_URL = os.environ.get("CLOUD_SERVE_URL", "")
LOCAL_TIMEOUT = float(os.environ.get("LOCAL_TIMEOUT", "10"))
CLOUD_TIMEOUT = float(os.environ.get("CLOUD_TIMEOUT", "300"))

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(CLOUD_TIMEOUT))
    return _client


async def _proxy_request(request: Request, target_url: str, source: str) -> Response:
    """Forward request to target URL, streaming the response back."""
    client = _get_client()

    url = f"{target_url}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers["x-forge-source"] = source

    body = await request.body()

    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )
    resp = await client.send(req, stream=True)

    response_headers = dict(resp.headers)
    response_headers["x-forge-source"] = source

    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=response_headers,
    )


async def overflow_proxy(request: Request) -> Response:
    """Try local Ray Serve, fall back to cloud on failure."""
    start = time.time()

    # Try local first
    try:
        resp = await _proxy_request(request, LOCAL_RAY_URL, "local")
        if resp.status_code != 503:
            elapsed = time.time() - start
            logger.info("LOCAL %s %s → %d (%.2fs)", request.method, request.url.path, resp.status_code, elapsed)
            return resp
        logger.warning("LOCAL %s → 503, falling back to cloud", request.url.path)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("LOCAL %s failed: %s, falling back to cloud", request.url.path, e)

    # No cloud endpoint configured
    if not CLOUD_SERVE_URL:
        return Response(
            content=b'{"error": "local overloaded, no cloud endpoint configured"}',
            status_code=503,
            media_type="application/json",
        )

    # Fall back to cloud
    try:
        resp = await _proxy_request(request, CLOUD_SERVE_URL, "cloud")
        elapsed = time.time() - start
        logger.info("CLOUD %s %s → %d (%.2fs)", request.method, request.url.path, resp.status_code, elapsed)
        return resp
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.error("CLOUD %s also failed: %s", request.url.path, e)
        return Response(
            content=b'{"error": "both local and cloud failed"}',
            status_code=502,
            media_type="application/json",
        )


async def health(request: Request) -> Response:
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "local_url": LOCAL_RAY_URL,
        "cloud_url": CLOUD_SERVE_URL or "not configured",
    })


def create_app() -> Starlette:
    routes = [
        Route("/health", health),
        Route("/{path:path}", overflow_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
    ]
    return Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8090)
