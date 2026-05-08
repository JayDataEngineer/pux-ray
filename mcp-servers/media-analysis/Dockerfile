# Media Analysis MCP Server — CPU mode (for k3s worker deployment)
# Build context: ../media-analysis-mcp/ (cloned by build_mcp.sh)
#
# Build: docker build -f mcp-servers/media-analysis/Dockerfile -t tech-noir/mcp-media-analysis:latest ./repos/media-analysis-mcp

ARG TORCH_VARIANT=cpu

# Builder stage
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ARG TORCH_VARIANT

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libffi-dev \
    libzbar-dev \
    libchromaprint-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
        uv pip install --python /app/.venv/bin/python \
        --force-reinstall \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    else \
        uv pip install --python /app/.venv/bin/python \
        --force-reinstall \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124; \
    fi

COPY src/ /app/src/

# Final stage
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libzbar0 \
    libchromaprint1 \
    libchromaprint-tools \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    HF_HOME=/app/models

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
