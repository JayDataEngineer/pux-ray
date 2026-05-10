#!/usr/bin/env bash
# Tag and push Docker images to GHCR for SkyServe cloud burst.
#
# Prerequisites:
#   1. docker login ghcr.io -u JayDataEngineer
#   2. Images already pushed to Forge Registry (bash infra/k8s/build_and_import.sh)
#
# Usage: bash infra/docker/push-images.sh
set -euo pipefail

FORGE_REGISTRY="100.86.69.57:30500"
GHCR_REGISTRY="ghcr.io/jaydataengineer/tech-noir"

declare -A IMAGES=(
    ["gpu-all"]="gpu-all"
)

echo "=== Pushing images to GHCR ==="

for name in "${!IMAGES[@]}"; do
    local_tag="${FORGE_REGISTRY}/tech-noir/${IMAGES[$name]}:latest"
    remote_tag="${GHCR_REGISTRY}/${name}:latest"

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
echo "Done. Images available at ${GHCR_REGISTRY}"
