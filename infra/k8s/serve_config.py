"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

ALL model services go through the Forge → wan2GP → family_handlers system.
No standalone service deployments. One system, one truth.
"""

# ─── The Forge — VRAM-aware GPU manager (Wan2GP + ComfyUI + LLM) ────────────
# Wan2GP registered as a forge service (vram_mb=0, mmgp self-manages).
# Forge's eviction logic handles model swapping across all services.
from services.forge import forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# API Ingress — catch-all gateway (/health, /status, /v1/*, /dashboard, /studio)
# LAST app — Ray Serve matches most-specific route_prefix first.
from gateway.ingress_deployment import APIIngressDeployment

api_ingress = APIIngressDeployment.bind()
