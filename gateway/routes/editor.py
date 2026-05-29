"""Video editor frontend — serves the built React SPA.

Serves static files from gateway/editor/ (built by web/editor/ via Vite).
The SPA calls /v1/wf API endpoints directly from the browser.
"""
from __future__ import annotations

import logging
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

logger = logging.getLogger(__name__)

_EDITOR_DIR = Path(__file__).resolve().parents[1] / "editor"

_MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


async def editor_page(request: Request) -> HTMLResponse:
    """GET /editor — serve the video editor SPA."""
    index = _EDITOR_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h1>Editor not built</h1><p>Run <code>cd web/editor && pnpm build</code></p>",
            status_code=404,
        )
    return HTMLResponse(
        index.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


async def editor_static(request: Request) -> Response:
    """GET /editor/{path} — serve static assets."""
    path = request.path_params.get("path", "")
    file_path = _EDITOR_DIR / path

    # Prevent directory traversal
    try:
        file_path = file_path.resolve()
        file_path.relative_to(_EDITOR_DIR.resolve())
    except ValueError:
        return Response(status_code=403)

    if not file_path.exists() or not file_path.is_file():
        # SPA fallback — serve index.html for client-side routing
        index = _EDITOR_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return Response(status_code=404)

    ext = file_path.suffix.lower()
    media_type = _MIME_TYPES.get(ext, "application/octet-stream")

    # Hashed asset filenames (e.g. index-AbCd123.js) are immutable
    is_hashed = len(file_path.stem.split("-")) > 1 and any(c.isdigit() for c in file_path.stem)
    cache = "public, immutable, max-age=31536000" if is_hashed else "no-cache"

    return Response(
        content=file_path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": cache},
    )
