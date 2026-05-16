# Production Concerns

## Cloud Burst — Model Storage for Horizontal Scaling

**Problem:** When SkyPilot spins up cloud instances, model weights need to be available
instantly. Auto-download from HuggingFace adds 3-10 min cold-start per model (16-33GB).

**Decision:** Use cloud provider network volumes (RunPod Network Storage, etc.) for model
persistence across instances. ~$0.10/GB/month. First instance downloads, rest are instant.

**Why not self-host (Garage):** Home server upload bandwidth (~100 Mbps) makes 16-33GB
model transfers take 20-45 minutes. Cloud network volumes are same-datacenter, multi-GB/s.
Self-hosted S3 is correct for local infra, wrong for cloud burst.

**Architecture:**
```
SkyServe Instance
├── Docker image (~15GB, code + CUDA exts only, no model weights)
├── /models  ← persistent network volume (shared across ALL instances)
│   ├── 3d/pixal3d/, 3d/trellis/, audio/moss-soundeffect/, ...
└── setup: python -m registry.cli pull $PREWARM_MODELS
```

**Configurable per-deployment:** `PREWARM_MODELS=pixal3d,pixal3d_dinov3` in serve.yaml.
Different model sets for different burst targets (AniGen burst, Pixal3D burst, etc.).

**Alternative considered:** Mirror models to Cloudflare R2 (free egress), cloud instances
pull from R2. Faster than home upload, slower than network volume, more moving parts.
Not worth it.

**Files:** `infra/skypilot/serve.yaml`, `config/model_registry.yaml`, `registry/cli.py`
