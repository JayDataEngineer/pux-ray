#!/usr/bin/env bash
# Start Ray head node
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAY_BIN="$PROJECT_DIR/.venv/bin/ray"

# Check if Ray is already running
if "$RAY_BIN" status 2>/dev/null | grep -q "node"; then
    echo "Ray cluster already running."
    echo "Dashboard: http://localhost:8265"
    exit 0
fi

echo "Starting Ray cluster..."

"$RAY_BIN" start --head \
    --num-cpus=16 \
    --num-gpus=1 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --object-store-memory=4000000000 \
    --temp-dir="/tmp/ray"

echo ""
echo "Ray cluster started."
echo "  Dashboard: http://localhost:8265"
echo "  Resources: 1 GPU, 16 CPUs"
echo ""
echo "To deploy services:"
echo "  cd $PROJECT_DIR && .venv/bin/python -m scripts.deploy_services"
