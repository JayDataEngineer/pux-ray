# Developing on a Constrained Cluster

Single RTX 4090 (24GB VRAM), single GPU worker pod. All services route through
the unified wan2gp → model_engine system.

## Architecture: One System, One Truth

Every model service — GPU and CPU — goes through a single path:

```
Request → Forge → Wan2GPService → model_engine handler → orchestrator → forward()
```

- **Forge** (`services/forge.py`): VRAM-aware GPU manager with 3 services: `wan2gp`, `comfyui`, `llm`
- **Wan2GPService** (`services/wan2gp/deployment.py`): Unified model pool with V2V_MODELS registry (17 entries)
- **model_engine handlers** (`services/model_engine/handlers/`): nn.Module decomposition + orchestrator
- **Vendor handlers** (wan, hunyuan, flux, ace_step, index_tts): Upstream Wan2GP family_handler code

### Handler pattern (3-file structure)

```
handlers/<family>/
  __init__.py      # BaseHandler implementation + variant metadata
  modules.py       # Load nn.Modules, build pipe dict, extract weights
  orchestrator.py  # Raw forward() calls — the inference logic
```

Exceptions (no nn.Modules):
- `espeak/` — subprocess binary, no modules.py
- `faster_whisper/` — CTranslate2 backend, no modules.py

## Service Tiers

### Tier 1 — Forge Services (always available via `/forge`)

| Service | GPU | VRAM |
|---------|-----|------|
| wan2gp | 1.0 (self-managed via mmgp) | varies by model |
| comfyui | subprocess | exclusive GPU |
| llm | subprocess | exclusive GPU |

### Models in V2V_MODELS (wan2gp pool)

| Model | Engine | VRAM |
|-------|--------|------|
| wan/t2v-14B | vendor | 14GB |
| wan/i2v-14B | vendor | 14GB |
| hunyuan/t2v | vendor | 12GB |
| flux/t2i | vendor | 8GB |
| ace_step/v1_5 | vendor | 8GB |
| index_tts/v2 | vendor | 6GB (blocked) |
| anigen | model_engine | 12GB |
| trellis | model_engine | 10GB |
| hy_motion | model_engine | 6GB |
| moss_soundeffect | model_engine | 16GB |
| see_through | model_engine | 6GB |
| vibevoice_asr | model_engine | 16GB |
| vibevoice_tts | model_engine | 18GB |
| kokoro | model_engine (CPU) | 0 |
| espeak | model_engine (CPU) | 0 |
| faster_whisper | model_engine (CPU) | 0 |

## Workflow: Debug a Model

### Step 1: Test locally

```bash
WORKER=$(kubectl get pods -n ai-services -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')

# Test through the unified system
kubectl exec -n ai-services $WORKER -c ray-worker -- python3 -c "
from services.wan2gp.deployment import Wan2GPService
svc = Wan2GPService()
svc.load('trellis')
result = svc.infer({'prompt': 'a cat', 'steps': 1, 'image_b64': 'x'})
print(result.get('status'))
svc.unload()
"
```

### Step 2: Fix the code

Handler code lives in `services/model_engine/handlers/<family>/`. The 3-file
structure means you can usually fix inference logic in `orchestrator.py`
without touching module loading.

```bash
# Rebuild + recycle
bash infra/k8s/build_and_import.sh
kubectl delete pods -n ai-services -l ray.io/is-ray-node=yes
```

### Step 3: Run the test suite

```bash
python tests/test_all_services.py
```

## GPU Budget

Total VRAM: 24GB. mmgp manages VRAM per-module, not per-model. Multiple
models can share the GPU through the mmgp pool with per-layer swapping.

Budgets in `services/wan2gp/deployment.py`:

```python
budgets = {"transformer": 250, "text_encoder": 250, "*": 3000}
```

## Blocked: IndexTTS via Wan2GP

IndexTTS through Wan2GP is blocked. Wan2GP vendors a forked copy of
`transformers/generation/utils.py` incompatible with transformers >=4.55.
Resolution: wait for Wan2GP to update, or shim the missing symbols.

## Adding a New Model

1. Create handler in `services/model_engine/handlers/<name>/`
2. Implement `modules.py` (nn.Module decomposition) and `orchestrator.py` (forward() calls)
3. Register in `V2V_MODELS` in `services/wan2gp/deployment.py`
4. Add test payload to `tests/test_all_services.py`

No changes needed to `serve_config.py`, `ray-service.yaml`, or the Forge.
