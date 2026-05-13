"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

ALL model services go through the Forge → wan2gp → model_engine system.
No standalone service deployments. One system, one truth.
"""

# ─── The Forge — VRAM-aware GPU manager (all models route here) ──────────────
from services.forge import forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# API Ingress — catch-all gateway (/health, /status, /v1/*, /dashboard, /studio)
# LAST app — Ray Serve matches most-specific route_prefix first.
from gateway.ingress_deployment import APIIngressDeployment

api_ingress = APIIngressDeployment.bind()

# ─── All models route through /forge → wan2gp V2V_MODELS ────────────────────
# GPU: wan/t2v, wan/i2v, hunyuan, flux, ace_step, index_tts, anigen, trellis,
#      hy_motion, moss, see_through, faster_qwen3_tts, vibevoice_asr, vibevoice_tts
# CPU: kokoro, espeak, faster_whisper
# See services/wan2gp/deployment.py V2V_MODELS for the full registry.
