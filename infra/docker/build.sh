#!/usr/bin/env bash
# Tech Noir Ray — Docker Image Builder
# =====================================
# Builds all creative tool Docker images for runtime_env["container"].
# Push to registry after building.
#
# Usage:
#   bash infra/docker/build.sh              # Build all
#   bash infra/docker/build.sh trellis      # Just TRELLIS
#   bash infra/docker/build.sh --push all   # Build + push to registry
#
# Images:
#   tech-noir/trellis     — TRELLIS.2 image-to-3D
#   tech-noir/anigen      — AniGen image-to-rigged-3D
#   tech-noir/seethrough  — See-Through layer decomposition
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REGISTRY="${DOCKER_REGISTRY:-tech-noir}"
PUSH=false

# Parse --push
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        *) ARGS+=("$arg") ;;
    esac
done
set -- "${ARGS[@]}"
TARGET="${1:-all}"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[build]${NC} $*"; }

build_image() {
    local name="$1" tag="$2" dockerfile="$3"
    local full_tag="${REGISTRY}/${name}:${tag}"
    log "Building ${full_tag}..."
    docker build \
        -f "${SCRIPT_DIR}/${dockerfile}" \
        -t "${full_tag}" \
        "${PROJECT_ROOT}" || { echo -e "${RED}FAILED: ${name}${NC}"; return 1; }
    log "Built: ${full_tag}"
    if $PUSH; then
        log "Pushing ${full_tag}..."
        docker push "${full_tag}"
    fi
}

case "$TARGET" in
    trellis)
        build_image trellis latest Dockerfile.trellis
        ;;
    anigen)
        build_image anigen latest Dockerfile.anigen
        ;;
    seethrough)
        build_image seethrough latest Dockerfile.seethrough
        ;;
    all)
        log "Building all Docker images..."
        build_image trellis latest Dockerfile.trellis &
        build_image anigen latest Dockerfile.anigen &
        build_image seethrough latest Dockerfile.seethrough &
        wait
        log "All images built."
        if $PUSH; then
            log "All images pushed to ${REGISTRY}."
        fi
        ;;
    *)
        echo "Usage: $0 [--push] {trellis|anigen|seethrough|all}"
        exit 1
        ;;
esac
