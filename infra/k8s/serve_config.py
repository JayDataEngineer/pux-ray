"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

ALL model services go through the Forge → wan2gp → model_engine system.
No standalone service deployments. One system, one truth.
"""

# ─── Wan2GP — unified GPU model deployment (all vendor + model_engine models) ──
# num_gpus: 1.0 — mmgp handles VRAM/CPU/RAM management for ALL models
# Dynamic registry: auto-discovers models from 19 vendor handlers + 12 model_engine
# See services/wan2gp/deployment.py for the full dynamic registry system.
from services.wan2gp.serve_deployment import wan2gp_deployment

# ─── The Forge — VRAM-aware GPU manager (ComfyUI + LLM subprocess services) ────
from services.forge import forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# API Ingress — catch-all gateway (/health, /status, /v1/*, /dashboard, /studio)
# LAST app — Ray Serve matches most-specific route_prefix first.
from gateway.ingress_deployment import APIIngressDeployment

api_ingress = APIIngressDeployment.bind()
