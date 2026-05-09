#!/usr/bin/env bash
# Tag and push Docker images to GHCR for SkyServe cloud burst.
#
# Prerequisites:
#   1. docker login ghcr.io -u JayDataEngineer
#   2. Local images already built (task setup or bash infra/k8s/build_and_import.sh)
#
# Usage: bash infra/docker/push-images.sh
set -euo pipefail

REGISTRY="ghcr.io/jaydataengineer/tech-noir"

declare -A IMAGES=(
    ["gpu-all"]="localhost/tech-noir/gpu-all:latest"
)

echo "=== Pushing images to GHCR ==="

for name in "${!IMAGES[@]}"; do
    local_tag="${IMAGES[$name]}"
    remote_tag="${REGISTRY}/${name}:latest"

    if ! docker inspect "$local_tag" > /dev/null 2>&1; then
        echo "SKIP: ${local_tag} not found locally"
        continue
    fi

    echo ""
    echo "--- Pushing ${name} ---"
    docker tag "$local_tag" "$remote_tag"
    docker push "$remote_tag"
    echo "OK: ${remote_tag}"
done

echo ""
echo "Done. Images available at ${REGISTRY}"
