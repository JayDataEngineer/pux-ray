#!/bin/bash
# Sidecar: auto-pull ComfyUI custom extension repos and reload if changed.
# Called by systemd timer (runs as 'user').
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_CUSTOM_NODES="/home/user/Documents/programs/ray/infra/repos/ComfyUI/custom_nodes"
LOG_FILE="/tmp/tech-noir/comfyui-sidecar.log"

mkdir -p "$(dirname "$LOG_FILE")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

reload_comfyui() {
    log "ComfyUI extension changed, reloading..."
    cd "$SCRIPT_DIR/.."
    .venv/bin/python -c "
import ray
from ray.serve.handle import DeploymentHandle
ray.init(address='auto', namespace='tech_noir')
try:
    h = DeploymentHandle('comfyui', 'comfyui')
    h.stop_comfyui.remote()
    log('ComfyUI stop triggered, will restart on next request')
except Exception as e:
    print(f'Failed to reload ComfyUI: {e}')
" 2>&1 | tee -a "$LOG_FILE" || log "ComfyUI reload attempt failed (may need manual restart)"
}

changed=0

for repo_dir in "$COMFYUI_CUSTOM_NODES"/*/; do
    [ -d "$repo_dir/.git" ] || continue
    repo_name=$(basename "$repo_dir")
    remote=$(git -C "$repo_dir" remote get-url origin 2>/dev/null || echo "unknown")

    log "Checking $repo_name ($remote)..."

    before=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo "unknown")
    output=$(git -C "$repo_dir" pull --ff-only origin master 2>&1) && rc=0 || rc=$?
    after=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo "unknown")

    if [ "$before" != "$after" ]; then
        log "  Updated $repo_name: $before -> $after"
        changed=1
    else
        log "  $repo_name up to date ($before)"
    fi
done

if [ "$changed" -eq 1 ]; then
    reload_comfyui
fi
