#!/usr/bin/env bash
# Tech Noir Ray — Creative Tool Venv Setup
# ==========================================
# Creates Python venvs for creative tools cloned by `docker compose run --rm <tool>-sync`.
# Idempotent — safe to re-run. Skips tools with existing working venvs.
#
# Usage:
#   bash infra/setup_venvs.sh          # Set up all tools
#   bash infra/setup_venvs.sh trellis  # Set up TRELLIS only
#   bash infra/setup_venvs.sh llama    # Build llama.cpp only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOS_DIR="$SCRIPT_DIR/repos"
UV="$(which uv 2>/dev/null || echo uv)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[venv]${NC} $*"; }
warn() { echo -e "${YELLOW}[venv]${NC} $*"; }
err()  { echo -e "${RED}[venv]${NC} $*"; }

# ─── TRELLIS.2 — image-to-3D mesh generation ─────────────────────────────────
setup_trellis() {
    local dir="$REPOS_DIR/TRELLIS.2"
    local venv_py="$dir/.venv/bin/python"
    if [ -x "$venv_py" ] && "$venv_py" -c "import torch; print(torch.__version__)" &>/dev/null; then
        log "TRELLIS.2 venv OK (Python $("$venv_py" --version 2>&1))"
        return 0
    fi
    if [ ! -f "$dir/setup.sh" ]; then
        warn "TRELLIS.2 not cloned yet. Run: docker compose run --rm tools-sync"
        return 1
    fi
    log "Setting up TRELLIS.2 venv (CUDA extensions: o-voxel)..."
    cd "$dir"
    bash setup.sh --new-env --basic --o-voxel
    log "TRELLIS.2 venv ready"
}

# ─── AniGen — animated 3D character generation ───────────────────────────────
setup_anigen() {
    local dir="$REPOS_DIR/AniGen"
    local venv_py="$dir/.venv/bin/python"
    if [ -x "$venv_py" ] && "$venv_py" -c "import torch; print(torch.__version__)" &>/dev/null; then
        log "AniGen venv OK (Python $("$venv_py" --version 2>&1))"
        return 0
    fi
    if [ ! -f "$dir/setup.sh" ]; then
        warn "AniGen not cloned yet. Run: docker compose run --rm tools-sync"
        return 1
    fi
    log "Setting up AniGen venv..."
    cd "$dir"
    bash setup.sh --new-env --all
    log "AniGen venv ready"
}

# ─── ACE-Step 1.5 — text-to-music generation ─────────────────────────────────
setup_ace_step() {
    local dir="$REPOS_DIR/ACE-Step-1.5"
    local venv_py="$dir/.venv/bin/python"
    if [ -x "$venv_py" ] && "$venv_py" -c "import torch; print(torch.__version__)" &>/dev/null; then
        log "ACE-Step venv OK (Python $("$venv_py" --version 2>&1))"
        return 0
    fi
    if [ ! -f "$dir/pyproject.toml" ]; then
        warn "ACE-Step not cloned yet. Run: docker compose run --rm tools-sync"
        return 1
    fi
    log "Setting up ACE-Step venv (uv sync)..."
    cd "$dir"
    $UV venv --python 3.12 --quiet
    $UV sync --extra-index-url https://download.pytorch.org/whl/cu128
    log "ACE-Step venv ready"
}

# ─── See-Through — anime character layer decomposition ────────────────────────
setup_see_through() {
    local dir="$REPOS_DIR/see-through"
    local venv_py="$dir/.venv/bin/python"
    if [ -x "$venv_py" ] && "$venv_py" -c "import torch; print(torch.__version__)" &>/dev/null; then
        log "see-through venv OK (Python $("$venv_py" --version 2>&1))"
        return 0
    fi
    if [ ! -f "$dir/requirements.txt" ]; then
        warn "see-through not cloned yet. Run: docker compose run --rm tools-sync"
        return 1
    fi
    log "Setting up see-through venv..."
    cd "$dir"
    $UV venv --python 3.12 --quiet
    source .venv/bin/activate
    $UV pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    $UV pip install -r requirements.txt
    log "see-through venv ready"
}

# ─── Qwen Image Expert (Qwen3-TTS LoRA Trainer) ──────────────────────────────
setup_qwen_img() {
    local dir="$REPOS_DIR/qwen_img_expert"
    local venv_py="$dir/.venv/bin/python"
    if [ -x "$venv_py" ] && "$venv_py" -c "import torch; print(torch.__version__)" &>/dev/null; then
        log "qwen_img_expert venv OK (Python $("$venv_py" --version 2>&1))"
        return 0
    fi
    if [ ! -f "$dir/pyproject.toml" ]; then
        warn "qwen_img_expert not cloned yet. Run: docker compose run --rm tools-sync"
        return 1
    fi
    log "Setting up qwen_img_expert venv (uv sync)..."
    cd "$dir"
    $UV venv --python 3.12 --quiet
    $UV sync
    log "qwen_img_expert venv ready"
}

# ─── llama.cpp — server build ────────────────────────────────────────────────
setup_llama() {
    local dir="$REPOS_DIR/llama.cpp"
    if [ -f "$dir/build/bin/llama-server" ]; then
        log "llama.cpp already built ($("$dir/build/bin/llama-server" --version 2>&1 | head -1))"
        return 0
    fi
    if [ ! -f "$dir/CMakeLists.txt" ]; then
        warn "llama.cpp not cloned yet. Run: docker compose run --rm llama-sync"
        return 1
    fi
    log "Building llama.cpp..."
    cd "$dir"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
    cmake --build build --config Release -j"$(nproc)"
    log "llama.cpp build ready ($(ls build/bin/llama-* 2>/dev/null | wc -l) binaries)"
}

# ─── Migrate existing venvs from old locations ───────────────────────────────
migrate_venv() {
    local old_dir="$1"
    local new_dir="$2"
    local name="$3"
    if [ -d "$old_dir/.venv" ] && [ ! -e "$new_dir/.venv" ]; then
        log "Migrating $name venv: $old_dir/.venv -> $new_dir/.venv"
        cp -a "$old_dir/.venv" "$new_dir/.venv"
    elif [ ! -d "$old_dir/.venv" ] && [ -d "$new_dir/.venv" ]; then
        log "$name venv already at new location"
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    local target="${1:-all}"

    case "$target" in
        trellis)
            [ -d "$REPOS_DIR/TRELLIS.2" ] || migrate_venv \
                "/home/ubuntu/Documents/programs/TRELLIS.2" "$REPOS_DIR/TRELLIS.2" "TRELLIS.2"
            setup_trellis
            ;;
        anigen)
            [ -d "$REPOS_DIR/AniGen" ] || migrate_venv \
                "/home/ubuntu/Documents/programs/AniGen" "$REPOS_DIR/AniGen" "AniGen"
            setup_anigen
            ;;
        ace-step)
            [ -d "$REPOS_DIR/ACE-Step-1.5" ] || migrate_venv \
                "/home/ubuntu/Documents/programs/vid/ACE-Step-1.5" "$REPOS_DIR/ACE-Step-1.5" "ACE-Step"
            setup_ace_step
            ;;
        see-through)
            [ -d "$REPOS_DIR/see-through" ] || migrate_venv \
                "/home/ubuntu/Documents/programs/creative/see-through" "$REPOS_DIR/see-through" "see-through"
            setup_see_through
            ;;
        qwen)
            [ -d "$REPOS_DIR/qwen_img_expert" ] || migrate_venv \
                "/home/ubuntu/Documents/programs/creative/qwen_img_expert" "$REPOS_DIR/qwen_img_expert" "qwen_img_expert"
            setup_qwen_img
            ;;
        llama)
            setup_llama
            ;;
        all)
            log "Setting up all creative tools..."
            for tool in trellis anigen ace-step see-through qwen llama; do
                echo ""
                "$0" "$tool" || warn "  $tool setup had issues (may be OK if already set up)"
            done
            echo ""
            log "All creative tool venvs + llama.cpp build complete."
            log "Update config/local.yaml paths to point at $REPOS_DIR/<tool>/.venv/bin/python"
            ;;
        *)
            echo "Usage: $0 {trellis|anigen|ace-step|see-through|qwen|llama|all}"
            exit 1
            ;;
    esac
}

main "$@"
