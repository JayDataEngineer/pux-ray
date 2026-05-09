# Web Research MCP Server (with Celery worker support)
# Build context: ../local-web-mcp/ (cloned by build_mcp.sh)
#
# Build: docker build -f mcp-servers/web-research/Dockerfile -t tech-noir/mcp-web-research:latest ./repos/local-web-mcp

# Builder stage
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxshmfence1 \
    libxkbfile1 \
    libasound2 \
    libasound2-data \
    libu2f-udev \
    libvulkan1 \
    fonts-liberation \
    fonts-noto-color-emoji \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
    --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN .venv/bin/playwright install --with-deps chromium

RUN apt-get update && apt-get install -y \
    libxss1 \
    libayatana-appindicator3-1 \
    xdg-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome*.deb && rm ./google-chrome*.deb

RUN .venv/bin/seleniumbase get chromedriver

COPY src/ /app/src/

# Final stage
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxshmfence1 \
    libxkbfile1 \
    libasound2 \
    fonts-liberation \
    fonts-noto-color-emoji \
    xvfb \
    curl \
    wget \
    libxss1 \
    libayatana-appindicator3-1 \
    xdg-utils \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome*.deb && \
    rm ./google-chrome*.deb

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /root/.cache /root/.cache

RUN mkdir -p /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONWARNINGS="ignore::RequestsDependencyWarning,ignore::DeprecationWarning"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.mcp_sse:app", "--host", "0.0.0.0", "--port", "8000"]
