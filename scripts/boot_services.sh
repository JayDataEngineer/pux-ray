#!/usr/bin/env bash
# Boot script for Tech Noir Ray cluster.
# Starts Ray, deploys services, launches MCP servers and ingress.
# Installed as a systemd service — runs automatically after LUKS unlock + boot.
set -euo pipefail

PROJECT_ROOT="/home/user/Documents/programs/ray"
VENV="$PROJECT_ROOT/.venv/bin"
LOG_DIR="/tmp/tech-noir"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait for network
log "Waiting for network..."
until ping -c 1 -W 2 8.8.8.8 &>/dev/null; do sleep 2; done
log "Network ready"

# Clean old Ray session logs from tmpfs to prevent disk exhaustion.
# Ray stores ~1-2 GB of logs per session; old sessions never auto-removed.
if [ -d /tmp/ray ]; then
    log "Cleaning old Ray session logs..."
    find /tmp/ray/session_* -maxdepth 0 -type d 2>/dev/null | while read d; do
        echo "  Removing $d"
        rm -rf "$d"
    done
fi

# Start Ray cluster
log "Starting Ray cluster..."
bash "$PROJECT_ROOT/scripts/start_cluster.sh" >> "$LOG_DIR/ray.log" 2>&1
log "Ray cluster started"

# Deploy services
log "Deploying Ray Serve services..."
"$VENV/python" -m scripts.deploy_services >> "$LOG_DIR/deploy.log" 2>&1
log "Services deployed"

# Start MCP servers
log "Starting MCP servers..."
bash "$PROJECT_ROOT/scripts/start_mcp.sh" start >> "$LOG_DIR/mcp.log" 2>&1
log "MCP servers started"

# Start ingress
log "Starting ingress on port 18080..."
cd "$PROJECT_ROOT"
nohup "$VENV/python" -c "
import ray
ray.init(address='auto', namespace='tech_noir')
import uvicorn
from gateway.ingress import create_app
uvicorn.run(create_app(), host='0.0.0.0', port=18080)
" >> "$LOG_DIR/ingress.log" 2>&1 &
log "Ingress PID: $!"

log "Boot complete. All services running."
log "  Ingress:    http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}'):18080"
log "  Dashboard:  http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}'):18080/dashboard"
log "  Ray:        http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}'):18265"
