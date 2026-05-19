# =============================================================================
# Media Analysis MCP Server — CPU and GPU modes
#
# Usage:
#   CPU (default):   docker compose up -d
#   GPU (NVIDIA):    MEDIA_DEVICE=cuda docker compose up -d
#
# The TORCH_VARIANT build arg controls which PyTorch gets installed.
# Set MEDIA_DEVICE=cuda in docker-compose to switch runtime device.
# =============================================================================

ARG TORCH_VARIANT=cpu

# =============================================================================
# Builder stage - UV for fast dependency installation
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG TORCH_VARIANT

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# System deps for building llama-cpp-python, ultralytics, pyzbar, chromaprint, etc.
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

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv

# Install dependencies from pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Install PyTorch — CPU-only or CUDA depending on TORCH_VARIANT
# uv sync pulls the default (CUDA) from PyPI; we override it here
# CPU:  ~190MB  |  CUDA:  ~2GB+ (but GPU inference is 5-50x faster)
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

# Copy source code
COPY src/ /app/src/

# =============================================================================
# Final stage
# =============================================================================
FROM python:3.12-slim-bookworm

# Runtime deps: ffmpeg for video/audio, libs for numpy/onnx/pyzbar/chromaprint/soundfile
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

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml /app/pyproject.toml

# Create model cache directory
RUN mkdir -p /app/models

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    HF_HOME=/app/models

EXPOSE 8001

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
