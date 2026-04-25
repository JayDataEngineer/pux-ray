#!/usr/bin/env bash
# Stop Ray cluster gracefully
set -euo pipefail

echo "Stopping Ray cluster..."
ray stop
echo "Ray cluster stopped."
