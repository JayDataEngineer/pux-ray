#!/usr/bin/env bash
# Rebuild all Ray worker images and push to the Forge Registry.
# Run on the server: bash infra/k8s/build_and_import.sh
#
# Prerequisites:
#   1. Forge Registry deployed: kubectl apply -f infra/k8s/shared/forge-registry.yaml
#   2. Traefik registry entrypoint: kubectl apply -f infra/k8s/traefik-config.yaml
#   3. K3s containerd configured: see REGISTRIES_YAML below
set -euo pipefail

REGISTRY="forge-reg/tech-noir"      # K3s containerd resolves via registries.yaml
PUSH_REGISTRY="100.86.69.57:30500"  # Traefik NodePort for host Docker push
REGISTRIES_YAML="/etc/rancher/k3s/registries.yaml"

# Unified architecture: ONE golden image for ALL GPU services
IMAGES=(
    ray-base
    gpu-all
    model-sync
)

# --- Pre-flight checks ---
echo "=== Pre-flight checks ==="

if ! curl -sf "http://${PUSH_REGISTRY}/v2/" > /dev/null 2>&1; then
    echo "ERROR: Forge Registry not reachable at ${PUSH_REGISTRY}"
    echo "  Deploy:  kubectl apply -f infra/k8s/shared/forge-registry.yaml"
    echo "  Traefik: kubectl apply -f infra/k8s/traefik-config.yaml"
    echo "  Verify:  kubectl get pods -n infra -l app=forge-registry"
    exit 1
fi

if [ ! -f "$REGISTRIES_YAML" ]; then
    echo "WARNING: ${REGISTRIES_YAML} not found — K3s containerd won't resolve 'forge-reg'"
    echo "Create it with:"
    echo "  sudo mkdir -p /etc/rancher/k3s"
    echo "  sudo tee ${REGISTRIES_YAML} <<'EOF'"
    echo "  mirrors:"
    echo "    \"forge-reg\":"
    echo "      endpoint:"
    echo "        - \"http://forge-registry.infra.svc.cluster.local:5000\""
    echo "  EOF"
    echo "  sudo systemctl restart k3s"
    echo ""
fi

echo "Registry: OK (${PUSH_REGISTRY})"
echo ""

# --- Build and push ---
echo "=== Building ${#IMAGES[@]} images ==="

for img in "${IMAGES[@]}"; do
    dockerfile="infra/docker/Dockerfile.${img}"
    if [ ! -f "$dockerfile" ]; then
        echo "SKIP: $dockerfile not found"
        continue
    fi
    push_tag="${PUSH_REGISTRY}/tech-noir/${img}:latest"
    echo ""
    echo "--- Building ${img} ---"
    docker build -f "$dockerfile" -t "$push_tag" . 2>&1 | tail -5
    echo "Pushing ${img}..."
    docker push "$push_tag"
    echo "OK: ${img}"
done

echo ""
echo "=== All images built and pushed ==="
echo "Deploying RayService..."
kubectl apply -f infra/k8s/ray-service.yaml
echo "Done. Watch: kubectl get pods -n ai-services -w"
