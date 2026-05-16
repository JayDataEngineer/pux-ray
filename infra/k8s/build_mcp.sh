#!/usr/bin/env bash
# Build MCP server images and push to the Forge Registry.
# Source lives in mcp/ directory (in-repo), no external clones needed.
#
# Usage: bash infra/k8s/build_mcp.sh
set -euo pipefail

REGISTRY="forge-reg.local:30500/tech-noir"      # K3s containerd resolves via registries.yaml
PUSH_REGISTRY="100.86.69.57:30500"  # Traefik NodePort for host Docker push
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Image build configs: dockerfile → (image_name)
declare -A MCP_IMAGES=(
    ["mcp/media-analysis/Dockerfile"]="mcp-media-analysis"
    ["mcp/web-research/Dockerfile"]="mcp-web-research"
)

cd "$PROJECT_ROOT"

echo "=== Building MCP images from local sources ==="

for dockerfile in "${!MCP_IMAGES[@]}"; do
    image_name="${MCP_IMAGES[$dockerfile]}"
    context_dir="$(dirname "$dockerfile")"
    push_tag="${PUSH_REGISTRY}/tech-noir/${image_name}:latest"

    if [ ! -f "$dockerfile" ]; then
        echo "SKIP: $dockerfile not found"
        continue
    fi

    echo ""
    echo "--- Building ${image_name} ---"
    echo "  Dockerfile: $dockerfile"
    echo "  Context:    $context_dir"
    docker build -f "$dockerfile" -t "$push_tag" "$context_dir" 2>&1 | tail -5
    echo "Pushing ${image_name}..."
    docker push "$push_tag"
    echo "OK: ${image_name}"
done

echo ""
echo "=== Deploying MCP services ==="
python3 infra/secrets_sync.py
kubectl apply -f infra/k8s/mcp/namespace.yaml
kubectl apply -f infra/k8s/mcp/pvcs.yaml
kubectl apply -f infra/k8s/mcp/web-research-deps.yaml
kubectl apply -f infra/k8s/mcp/web-research-worker.yaml
kubectl apply -f infra/k8s/mcp/web-research.yaml
kubectl apply -f infra/k8s/mcp/media-analysis.yaml
kubectl apply -f infra/k8s/traefik-ingress.yaml

# ── Equibles (vendor clone, builds from upstream source) ──
EQUIBLES_DIR="vendor/equibles"
if [ -d "$EQUIBLES_DIR" ]; then
    echo ""
    echo "=== Building Equibles images from vendor clone ==="

    echo "--- Building equibles-mcp ---"
    docker build -f "$EQUIBLES_DIR/src/Equibles.Mcp.Server/Dockerfile" \
        -t "${PUSH_REGISTRY}/tech-noir/mcp-equibles:latest" "$EQUIBLES_DIR"
    docker push "${PUSH_REGISTRY}/tech-noir/mcp-equibles:latest"

    echo "--- Building equibles-worker ---"
    docker build -f "$EQUIBLES_DIR/src/Equibles.Worker.Host/Dockerfile" \
        -t "${PUSH_REGISTRY}/tech-noir/mcp-equibles-worker:latest" "$EQUIBLES_DIR"
    docker push "${PUSH_REGISTRY}/tech-noir/mcp-equibles-worker:latest"

    echo "--- Building equibles-web (migration runner) ---"
    docker build -f "$EQUIBLES_DIR/src/Equibles.Web/Dockerfile" \
        -t "${PUSH_REGISTRY}/tech-noir/mcp-equibles-web:latest" "$EQUIBLES_DIR"
    docker push "${PUSH_REGISTRY}/tech-noir/mcp-equibles-web:latest"

    echo "Deploying Equibles..."
    kubectl apply -f infra/flux/mcp/equibles-deps.yaml
    kubectl apply -f infra/flux/mcp/equibles-mcp.yaml
    kubectl apply -f infra/flux/mcp/equibles-worker.yaml
    kubectl apply -f infra/flux/mcp/equibles-web.yaml
    kubectl apply -f infra/flux/shared/keda.yaml
    kubectl apply -f infra/flux/mcp/equibles-scaledobject.yaml
fi

echo ""
echo "Done. Watch: kubectl get pods -n mcp -w"
