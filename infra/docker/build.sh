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
#   tech-noir/comfyui     — ComfyUI with flash-attn + extensions
#   tech-noir/trellis     — TRELLIS.2 image-to-3D (original)
#   tech-noir/trellis-spz — TRELLIS StableProjectorz (VRAM-optimized, 8GB)
#   tech-noir/anigen      — AniGen image-to-rigged-3D
#   tech-noir/vibevoice   — VibeVoice long-form TTS
#   tech-noir/seethrough  — See-Through layer decomposition
#   tech-noir/acestep     — ACE-STEP text-to-music
#   tech-noir/hymotion    — HY-Motion 1.0 text-to-3D human motion
#   tech-noir/gptsovits   — GPT-SoVITS voice cloning TTS
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
    comfyui)
        build_image comfyui latest Dockerfile.comfyui
        ;;
    trellis)
        build_image trellis latest Dockerfile.trellis
        ;;
    trellis-spz)
        build_image trellis-spz latest Dockerfile.trellis-spz
        ;;
    anigen)
        build_image anigen latest Dockerfile.anigen
        ;;
    vibevoice)
        build_image vibevoice latest Dockerfile.vibevoice
        ;;
    seethrough)
        build_image seethrough latest Dockerfile.seethrough
        ;;
    acestep)
        build_image acestep latest Dockerfile.acestep
        ;;
    gptsovits)
        build_image gptsovits latest Dockerfile.gptsovits
        ;;
    hymotion)
        build_image hymotion latest Dockerfile.hymotion
        ;;
    all)
        log "Building all Docker images..."
        build_image comfyui latest Dockerfile.comfyui &
        build_image trellis-spz latest Dockerfile.trellis-spz &
        build_image anigen latest Dockerfile.anigen &
        build_image vibevoice latest Dockerfile.vibevoice &
        build_image seethrough latest Dockerfile.seethrough &
        build_image acestep latest Dockerfile.acestep &
        build_image gptsovits latest Dockerfile.gptsovits &
        build_image hymotion latest Dockerfile.hymotion &
        wait
        log "All images built."
        if $PUSH; then
            log "All images pushed to ${REGISTRY}."
        fi
        ;;
    *)
        echo "Usage: $0 [--push] {comfyui|trellis|trellis-spz|anigen|vibevoice|seethrough|acestep|gptsovits|hymotion|all}"
        exit 1
        ;;
esac
