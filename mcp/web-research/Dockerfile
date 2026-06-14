# =============================================================================
# Builder stage - UV for fast dependency installation
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies needed for browsers
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

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv

# Install dependencies from pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Force-install PyTorch CPU-only AFTER uv sync.
# uv sync pulls CUDA variants from PyPI; this replaces them with CPU-only (~190MB vs 2GB+).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
    --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Playwright Chromium to /app/.playwright (NOT /app/.cache — the
# deployment mounts an emptyDir at /app/.cache which would shadow image-baked
# browsers, making them invisible at runtime).
RUN mkdir -p /app/.cache /app/.playwright && \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright \
    .venv/bin/playwright install --with-deps chromium

# Install Google Chrome for SeleniumBase fallback
# Note: libindicator7 is obsolete in Bookworm, using libayatana-appindicator instead
RUN apt-get update && apt-get install -y \
    libxss1 \
    libayatana-appindicator3-1 \
    xdg-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

RUN apt-get install -y ./google-chrome*.deb && rm ./google-chrome*.deb

# Install SeleniumBase chromedriver for Google Chrome
RUN .venv/bin/seleniumbase get chromedriver

# Copy source code
COPY src/ /app/src/

# =============================================================================
# Final stage
# =============================================================================
FROM python:3.13-slim-bookworm

# Install runtime dependencies only
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

# Install Google Chrome for SeleniumBase fallback
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome*.deb && \
    rm ./google-chrome*.deb

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY docs_config.yaml /app/docs_config.yaml
COPY profiling/ /app/profiling/
COPY tests/ /app/tests/
COPY pyproject.toml /app/pyproject.toml
# Copy Playwright browser cache (at /app/.playwright — outside the emptyDir mount)
COPY --from=builder /app/.playwright /app/.playwright

# Create data directory
RUN mkdir -p /app/data

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONWARNINGS="ignore::RequestsDependencyWarning,ignore::DeprecationWarning" \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright

EXPOSE 8000

CMD ["uvicorn", "src.mcp_sse:app", "--host", "0.0.0.0", "--port", "8000"]
