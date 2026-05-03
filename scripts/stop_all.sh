#!/usr/bin/env bash
# Stop Ray cluster gracefully and clean up session logs
set -euo pipefail

echo "Stopping Ray cluster..."
ray stop
echo "Ray cluster stopped."

# Clean old session logs from tmpfs to prevent disk exhaustion.
# Ray stores ~1-2 GB of logs per session in /tmp/ray/session_*/logs/.
if [ -d /tmp/ray ]; then
    echo "Cleaning Ray session logs from tmpfs..."
    rm -rf /tmp/ray/session_*
    echo "Done."
fi
