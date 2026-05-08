# Developing on a Constrained Cluster

Single RTX 4090 (24GB VRAM), single GPU worker pod. Tier 1 services are always running.
This guide covers how to develop and debug Tier 3 services without disrupting production.

## Production State (don't touch carelessly)

10 Tier 1 services are deployed and tested:

| Service | GPU | VRAM |
|---------|-----|------|
| kokoro, espeak, faster_whisper | CPU | 0 |
| faster_qwen3_tts | 0.20 | ~3.5GB |
| index_tts | 0.10 | ~1.5GB |
| vibevoice_cpp | 0.10 | ~1.5GB |
| ace_step | 0.40 | ~8GB |
| trellis | 0.95 | ~18GB |
| comfyui | 1.00 | variable |

Ray Serve autoscaling has `min_replicas: 0` on all GPU services — they load on first
request and unload after 30s idle. GPU fractions are soft reservations, not hard limits.
Multiple services can coexist if their combined VRAM fits in 24GB.

## Workflow: Fix a Tier 3 Service

### Step 1: Reproduce the failure

Exec into the running GPU worker and test the service code directly:

```bash
WORKER=$(kubectl get pods -n ai-services -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n ai-services $WORKER -c ray-worker -- python3 -c "
from services.tts.gpt_sovits import GPTSoVITSDeployment
d = GPTSoVITSDeployment()
d._load()
print('loaded ok')
"
```

This runs the service code in the same environment (Python, deps, models) as production
but without going through Ray Serve. If `_load()` fails, you see the real error immediately.

For subprocess-based services (ComfyUI, vibevoice.cpp), test the binary directly:

```bash
kubectl exec -n ai-services $WORKER -c ray-worker -- python3 -c "
import subprocess
proc = subprocess.Popen(
    ['python3', 'main.py', '--port', '18465', '--listen', '0.0.0.0'],
    cwd='/opt/ComfyUI',
    stdout=open('/tmp/comfyui-test.log', 'w'),
    stderr=subprocess.STDOUT,
)
import time; time.sleep(30)
proc.kill()
print(open('/tmp/comfyui-test.log').read()[-2000:])
"
```

### Step 2: Fix the code

Edit the service file on the host at `/home/user/Documents/programs/ray/services/...`.
Then rebuild the image and recycle the pods:

```bash
bash infra/k8s/build_and_import.sh
kubectl delete pods -n ai-services -l ray.io/is-ray-node=yes
```

Wait ~90 seconds for pods to come back up. Check with:

```bash
kubectl get pods -n ai-services -l ray.io/is-ray-node=yes
```

### Step 3: Deploy to Serve (one service at a time)

Uncomment the service in two files:

**`infra/k8s/serve_config.py`** — the Python import:
```python
# Uncomment:
from services.tts.gpt_sovits import GPTSoVITSDeployment
gpt_sovits = GPTSoVITSDeployment.bind()
```

**`infra/k8s/ray-service.yaml`** — the Serve config:
```yaml
- name: gpt_sovits
  import_path: serve_config:gpt_sovits
  route_prefix: /tts/gpt-sovits
  deployments:
    - name: gpt_sovits
      autoscaling_config: { min_replicas: 0, max_replicas: 1, downscale_delay_s: 30 }
      ray_actor_options: { num_gpus: 0.25 }
```

Then rebuild + recycle (Step 2). The service registers its route but doesn't load
until a request arrives (`min_replicas: 0`). If it crashes on load, only that
deployment fails — the other 10 keep serving.

### Step 4: Test

```bash
# Port-forward to the Ray Serve proxy
HEAD=$(kubectl get pods -n ai-services -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward --address 0.0.0.0 -n ai-services $HEAD 18080:8000 &

# Hit the new service
curl -s http://localhost:18080/tts/gpt-sovits/ \
  -d '{"action":"generate","input":{"text":"Hello"}}' --max-time 300
```

Run the full Tier 1 test suite to verify you didn't break anything:

```bash
.venv/bin/python scripts/test_services_v2.py --tier1 --timeout 300
```

### Step 5: Promote to Tier 1

If the service passes reliably (3+ runs, no timeouts), move it to the Tier 1 section
in `serve_config.py` and `ray-service.yaml`. Update `scripts/test_services_v2.py` to
include it in the default Tier 1 test run. Update `CLAUDE.md`.

## GPU Budget

Total VRAM: 24GB. Current Tier 1 GPU allocations sum to ~2.75 fractional, but actual
VRAM usage depends on which services are loaded simultaneously. Ray Serve autoscaling
means services share the GPU over time.

If a Tier 3 service needs significant VRAM, you may need to temporarily comment out a
Tier 1 GPU service (e.g. trellis at 0.95) to make room. This is safe — just uncomment
it back when done debugging.

**ComfyUI** is special: it uses `num_gpus: 1.0` (exclusive). When ComfyUI is loaded,
no other GPU service can start. It's safe because `min_replicas: 0` means it only loads
on demand, and Ray won't schedule it until all other GPU replicas are at zero.

## Useful Commands

```bash
# Cluster status
kubectl get pods -n ai-services -l ray.io/is-ray-node=yes -o wide

# Check which Serve routes are registered
kubectl exec -n ai-services $HEAD -c ray-head -- python3 -c "
import urllib.request; print(urllib.request.urlopen('http://localhost:8000/-/routes').read().decode())
"

# Check deployment health
kubectl exec -n ai-services $HEAD -c ray-head -- python3 -c "
import urllib.request, json
data = json.loads(urllib.request.urlopen('http://localhost:8265/api/serve/applications/').read())
for name, app in data.get('applications', {}).items():
    for dname, dep in app.get('deployments', {}).items():
        status = dep.get('status', '?')
        gpus = dep.get('deployment_config', {}).get('ray_actor_options', {}).get('num_gpus', 0)
        print(f'{dname:30s} {status:15s} GPUs={gpus}')
"

# Worker logs (service crashes show up here)
kubectl logs -n ai-services $WORKER -c ray-worker --tail=100

# Specific service replica logs
kubectl exec -n ai-services $WORKER -- find /tmp/ray -name "*.log" -path "*SERVICENAME*"
kubectl exec -n ai-services $WORKER -- tail -50 /tmp/ray/.../replica_name.log

# Subprocess logs (for SubprocessProxyMixin services)
kubectl exec -n ai-services $WORKER -- cat /tmp/tech-noir/subprocess.log

# Rebuild and redeploy (full cycle)
bash infra/k8s/build_and_import.sh && kubectl delete pods -n ai-services -l ray.io/is-ray-node=yes

# Run integration tests
.venv/bin/python scripts/test_services_v2.py --tier1 --timeout 300
.venv/bin/python scripts/test_services_v2.py --all --timeout 300
```

## Patterns

### Service won't load (import error, missing dep)

1. `kubectl exec` into worker, try the import manually
2. If the dep is missing from the image, add to `Dockerfile.gpu-all`
3. Rebuild + recycle

### Service loads but request fails (API incompat, wrong model)

1. Check the replica log for the traceback
2. Fix the `_generate()` method in the service file
3. Rebuild + recycle (code changes need a new image)

### Service times out on first request

This is normal for heavy GPU models. The `_load()` method has to:
- Download/allocate model weights (~seconds to minutes)
- Warm up CUDA kernels

If it consistently times out at 300s, increase the timeout in the test or
set `min_replicas: 1` to keep it warm (uses more VRAM).

### Service OOMs the GPU

1. Reduce `num_gpus` fraction (doesn't actually limit VRAM, just scheduling)
2. Add `torch.cuda.empty_cache()` in `_unload()`
3. Use quantized models (GGUF, GPTQ, AWQ)
4. Comment out a competing Tier 1 service while debugging
