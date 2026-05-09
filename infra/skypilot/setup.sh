#!/usr/bin/env bash
# Install SkyPilot and verify cloud provider access.
#
# Usage: bash infra/skypilot/setup.sh
set -euo pipefail

echo "=== Installing SkyPilot ==="
uv pip install "skypilot[runpod,lambda,kubernetes]"

echo ""
echo "=== Checking cloud access ==="
sky check

echo ""
echo "=== GPU price snapshot ==="
sky show-gpus RTX4090 A10G L4 A100 --all-regions

echo ""
echo "Done. Next steps:"
echo "  1. Add cloud API keys to config/secrets.env (RUNPOD_API_KEY, etc.)"
echo "  2. Push images:  task cloud:push"
echo "  3. Launch cloud: task cloud:up"
