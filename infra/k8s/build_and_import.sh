#!/usr/bin/env bash
# Rebuild all Ray worker images with updated code and import into k0s.
# Run on the server: bash infra/k8s/build_and_import.sh
set -euo pipefail

REGISTRY="localhost/tech-noir"
BUILDER="buildkit"

# All GPU service images that need rebuilding (code baked in via COPY services/)
IMAGES=(
    gpu-services
    comfyui
    trellis-spz
    hymotion
    anigen
    acestep
    seethrough
    gptsovits
    qwen-tts
    vibevoice
    moss-sfx
    tangoflux
    florence2
    phi4mm
)

echo "=== Building ${#IMAGES[@]} images ==="

for img in "${IMAGES[@]}"; do
    dockerfile="infra/docker/Dockerfile.${img}"
    if [ ! -f "$dockerfile" ]; then
        echo "SKIP: $dockerfile not found"
        continue
    fi
    tag="${REGISTRY}/${img}:latest"
    echo ""
    echo "--- Building ${img} ---"
    docker build -f "$dockerfile" -t "$tag" . 2>&1 | tail -5
    echo "Saving ${img}..."
    docker save "$tag" | sudo k0s ctr images import -
    echo "OK: ${img}"
done

echo ""
echo "=== All images built and imported ==="
echo "Deploying RayService..."
kubectl apply -f infra/k8s/ray-service.yaml
echo "Done. Watch: kubectl get pods -n ai-services -w"
