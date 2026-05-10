#!/usr/bin/env bash
# Build MCP server images and push to the Forge Registry.
# Clones/updates source repos from GitHub, builds Docker images, pushes to registry.
#
# Usage: bash infra/k8s/build_mcp.sh
set -euo pipefail

REGISTRY="forge-reg/tech-noir"      # K3s containerd resolves via registries.yaml
PUSH_REGISTRY="100.86.69.57:30500"  # Traefik NodePort for host Docker push
REPO_DIR="infra/repos"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source repos for MCP servers
declare -A MCP_REPOS=(
    ["media-analysis-mcp"]="https://github.com/JayDataEngineer/media-analysis-mcp.git"
    ["local-web-mcp"]="https://github.com/JayDataEngineer/local-web-mcp.git"
)

# Image build configs: dockerfile → (context_dir, image_name)
declare -A MCP_IMAGES=(
    ["mcp-servers/media-analysis/Dockerfile"]="media-analysis-mcp mcp-media-analysis"
    ["mcp-servers/web-research/Dockerfile"]="local-web-mcp mcp-web-research"
)

cd "$PROJECT_ROOT"

echo "=== Syncing MCP source repos ==="
mkdir -p "$REPO_DIR"

for repo_name in "${!MCP_REPOS[@]}"; do
    repo_url="${MCP_REPOS[$repo_name]}"
    repo_path="$REPO_DIR/$repo_name"

    if [ ! -d "$repo_path/.git" ]; then
        echo "Cloning $repo_name..."
        git clone --depth 1 "$repo_url" "$repo_path"
    else
        echo "Updating $repo_name..."
        git -C "$repo_path" pull --ff-only || echo "WARNING: Could not update $repo_name"
    fi
done

echo ""
echo "=== Building MCP images ==="

for dockerfile in "${!MCP_IMAGES[@]}"; do
    IFS=' ' read -r context_dir image_name <<< "${MCP_IMAGES[$dockerfile]}"
    context_path="$REPO_DIR/$context_dir"
    push_tag="${PUSH_REGISTRY}/tech-noir/${image_name}:latest"

    if [ ! -d "$context_path" ]; then
        echo "SKIP: $context_path not found (repo sync failed?)"
        continue
    fi

    echo ""
    echo "--- Building ${image_name} ---"
    docker build -f "$dockerfile" -t "$push_tag" "$context_path" 2>&1 | tail -5
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

echo ""
echo "Done. Watch: kubectl get pods -n mcp -w"
