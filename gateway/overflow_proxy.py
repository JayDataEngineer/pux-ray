"""Overflow Proxy — local Ray Serve with cloud fallback.

Forwards requests to local Ray Serve. If the local instance returns
503 (overloaded) or times out, falls back to CLOUD_SERVE_URL.

Environment:
  LOCAL_RAY_URL   — local Ray Serve (default: http://localhost:8000)
  CLOUD_SERVE_URL — cloud fallback (optional — if unset, proxy only locally)
  LOCAL_TIMEOUT   — seconds before treating local as failed (default: 10)
"""
from __future__ import annotations

import os
import time
import logging

import httpx
from prometheus_client import Counter, Histogram, generate_latest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

logger = logging.getLogger(__name__)

LOCAL_RAY_URL = os.environ.get("LOCAL_RAY_URL", "http://localhost:8000")
CLOUD_SERVE_URL = os.environ.get("CLOUD_SERVE_URL", "")
LOCAL_TIMEOUT = int(os.environ.get("LOCAL_TIMEOUT", "10"))

overflow_requests_total = Counter(
    "overflow_requests_total",
    "Total requests processed",
    ["backend", "status"],
)
overflow_request_duration_seconds = Histogram(
    "overflow_request_duration_seconds",
    "Request duration in seconds",
    ["backend"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)


async def _forward(request: Request, target_url: str, timeout: float) -> httpx.Response:
    url = f"{target_url}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k != "host"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.request(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
        )


async def proxy_handler(request: Request) -> Response:
    start = time.monotonic()

    # Try local first
    try:
        resp = await _forward(request, LOCAL_RAY_URL, LOCAL_TIMEOUT)
        overflow_request_duration_seconds.labels(backend="local").observe(time.monotonic() - start)

        if resp.status_code != 503:
            overflow_requests_total.labels(backend="local", status=str(resp.status_code)).inc()
            return Response(content=resp.content, status_code=resp.status_code)

        logger.info("Local returned 503 for %s, falling back to cloud", request.url.path)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        overflow_request_duration_seconds.labels(backend="local").observe(time.monotonic() - start)
        overflow_requests_total.labels(backend="local", status="timeout").inc()
        logger.warning("Local failed for %s: %s", request.url.path, e)

    # Cloud fallback
    if not CLOUD_SERVE_URL:
        return Response(
            content=b'{"error": "local unavailable, no cloud fallback configured"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        resp = await _forward(request, CLOUD_SERVE_URL, 120.0)
        overflow_request_duration_seconds.labels(backend="cloud").observe(time.monotonic() - start)
        overflow_requests_total.labels(backend="cloud", status=str(resp.status_code)).inc()
        return Response(content=resp.content, status_code=resp.status_code)
    except Exception as e:
        overflow_request_duration_seconds.labels(backend="cloud").observe(time.monotonic() - start)
        overflow_requests_total.labels(backend="cloud", status="error").inc()
        logger.error("Cloud fallback failed: %s", e)
        return Response(
            content=b'{"error": "both local and cloud unavailable"}',
            status_code=502,
            media_type="application/json",
        )


async def health(request: Request) -> Response:
    return Response(content=b'{"status":"ok"}', media_type="application/json")


async def metrics(request: Request) -> Response:
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def create_app() -> Starlette:
    return Starlette(routes=[
        Route("/health", health),
        Route("/metrics", metrics),
        Route("/{path:path}", proxy_handler, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ])
