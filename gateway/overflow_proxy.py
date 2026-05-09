from __future__ import annotations

import logging
import os
import time
from typing import Dict

import httpx
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    REGISTRY,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# ── Defaults from env ──────────────────────────────────────────────
LOCAL_RAY_URL = os.environ.get("LOCAL_RAY_URL", "http://ray-serve-proxy.ai-services.svc.cluster.local:8000")
CLOUD_SERVE_URL = os.environ.get("CLOUD_SERVE_URL", "")
LOCAL_TIMEOUT_DEFAULT = float(os.environ.get("LOCAL_TIMEOUT", "10"))
CLOUD_TIMEOUT_DEFAULT = float(os.environ.get("CLOUD_TIMEOUT", "300"))

# In-memory runtime config — overridable via POST /config
_config: Dict[str, float] = {
    "local_timeout": LOCAL_TIMEOUT_DEFAULT,
    "cloud_timeout": CLOUD_TIMEOUT_DEFAULT,
}

# ── Prometheus metrics ─────────────────────────────────────────────
METRIC_PREFIX = "overflow"

requests_total = Counter(
    f"{METRIC_PREFIX}_requests_total",
    "Total proxy requests by source and status",
    ["source", "status"],
)
request_duration = Histogram(
    f"{METRIC_PREFIX}_request_duration_seconds",
    "Request latency by source",
    ["source"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)
cloud_enabled = Gauge(
    f"{METRIC_PREFIX}_cloud_enabled",
    "1 if cloud endpoint is configured, 0 otherwise",
)

def _classify(status: int) -> str:
    if status < 400:
        return "2xx"
    if status < 500:
        return "4xx"
    return "5xx"

_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        t = _config["cloud_timeout"]
        _client = httpx.AsyncClient(timeout=httpx.Timeout(t))
    return _client

def _reset_client() -> None:
    global _client
    if _client is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_client.aclose())
            else:
                loop.run_until_complete(_client.aclose())
        except RuntimeError:
            pass
        _client = None

# ── Request tracing middleware ────────────────────────────────────
class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call):
        start = time.time()
        response = await call(request)
        elapsed = time.time() - start
        src = response.headers.get("x-forge-source", "unknown")
        logger.info(
            "%s %s %s → %d (%.2fs)",
            src.upper(), request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response

# ── Handlers ──────────────────────────────────────────────────────
async def overflow_proxy(request: Request) -> Response:
    start = time.time()
    gte = _get_client

    try:
        resp = await _proxy_request(request, LOCAL_RAY_URL, "local")
        if resp.status_code != 503:
            elapsed = time.time() - start
            requests_total.labels(source="local", status=_classify(resp.status_code)).inc()
            request_duration.labels(source="local").observe(elapsed)
            return resp
        logger.warning("LOCAL %s → 503, falling back to cloud", request.url.path)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("LOCAL %s failed: %s, falling back to cloud", request.url.path, e)

    if not CLOUD_SERVE_URL:
        elapsed = time.time() - start
        requests_total.labels(source="local", status="5xx").inc()
        request_duration.labels(source="local").observe(elapsed)
        return Response(
            content=b'{"error": "local overloaded, no cloud endpoint configured"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        resp = await _proxy_request(request, CLOUD_SERVE_URL, "cloud")
        elapsed = time.time() - start
        requests_total.labels(source="cloud", status=_classify(resp.status_code)).inc()
        request_duration.labels(source="cloud").observe(elapsed)
        return resp
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.error("CLOUD %s also failed: %s", request.url.path, e)
        requests_total.labels(source="cloud", status="5xx").inc()
        return Response(
            content=b'{"error": "both local and cloud failed"}',
            status_code=502,
            media_type="application/json",
        )

async def _proxy_request(request: Request, target_url: str, source: str) -> Response:
    client = _get_client()

    url = f"{target_url}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers["x-forge-source"] = source
    body = await request.body()

    req = client.build_request(
        method=request.method, url=url,
        headers=headers, content=body,
    )
    resp = await client.send(req, stream=True)

    response_headers = dict(resp.headers)
    response_headers["x-forge-source"] = source
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=response_headers,
    )

async def health(request: Request) -> Response:
    return JSONResponse({
        "status": "ok",
        "local_url": LOCAL_RAY_URL,
        "cloud_url": CLOUD_SERVE_URL or "not configured",
        "config": _config,
    })

async def handle_metrics(request: Request) -> Response:
    return Response(
        content=generate_latest(REGISTRY).decode(),
        media_type="text/plain; version=0.0.4",
    )

async def handle_config(request: Request) -> Response:
    if request.method == "GET":
        return JSONResponse({
            "local_timeout": _config["local_timeout"],
            "cloud_timeout": _config["cloud_timeout"],
            "cloud_url": CLOUD_SERVE_URL or None,
            "local_url": LOCAL_RAY_URL,
        })

    body = await request.json()
    changed: list[str] = []

    for key in ("local_timeout", "cloud_timeout"):
        if key in body:
            val = float(body[key])
            if val <= 0:
                return JSONResponse({"error": f"{key} must be > 0"}, status_code=400)
            _config[key] = val
            changed.append(key)

    if "cloud_timeout" in changed:
        _reset_client()

    return JSONResponse({
        "changed": changed,
        "config": {**_config},
        "cloud_url": CLOUD_SERVE_URL or None,
    })

# ── App factory ───────────────────────────────────────────────────
def create_app() -> Starlette:
    cloud_enabled.set(1 if CLOUD_SERVE_URL else 0)

    routes = [
        Route("/health", health),
        Route("/metrics", handle_metrics),
        Route("/config", handle_config, methods=["GET", "POST"]),
        Route("/{path:path}", overflow_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
    ]
    middleware = [Middleware(RequestLogMiddleware)]
    return Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8090)
