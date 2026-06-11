# Overflow Gateway — lightweight HTTP proxy for local/cloud fallback
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir uvicorn[standard] httpx starlette prometheus-client

# Copy gateway code
COPY gateway/overflow_proxy.py /app/gateway/

# Create a simple wrapper
RUN echo 'from gateway.overflow_proxy import create_app\nimport uvicorn\nuvicorn.run(create_app(), host="0.0.0.0", port=8090)' > /app/main.py

# Create non-root user
RUN groupadd -r appuser && useradd -r -u 1000 -g appuser appuser || true
RUN chown -R appuser:appuser /app

USER 1000
ENV PYTHONUNBUFFERED=1
EXPOSE 8090

CMD ["python", "-m", "uvicorn", "main:create_app", "--host", "0.0.0.0", "--port", "8090", "--factory"]
