#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Tech Noir Model Download Suite — IaC-driven from model_registry.yaml
# ════════════════════════════════════════════════════════════════════════════
# Reads model_registry.yaml to get model sources and downloads them.
# Organized into sections for targeted downloads.
#
# Usage:
#   ./models.sh                          # download ALL auto-downloadable models
#   ./models.sh --section audio           # only audio models
#   ./models.sh --list-sections           # list available sections
#   ./models.sh --list-models             # list all models with status
#   ./models.sh --list-missing            # list only missing auto-downloadable models
#   ./models.sh --list-manual             # list models requiring manual setup
#   ./models.sh --dry-run                 # show what would download, don't execute
#
# Special operations:
#   --fp8-qwen        Build Qwen-Image-Edit FP8 weight-only
#   --fp8-zimage      Build Z-Image Turbo/Base FP8
#   --fp8-vace        Build Wan VACE 14B FP8
#   --moss-gguf       Download MOSS GGUF models + ONNX tokenizer
#   --ace-xl          Download ACE-Step XL GGUF variants
#
# Design:
#   This script reads model_registry.yaml (the single source of truth)
#   to get download sources. If a model isn't in the registry, it can't
#   be downloaded through this script. Add it to the registry first.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REGISTRY="$PROJECT_ROOT/config/model_registry.yaml"
MODELS_ROOT="${MODELS_ROOT:-/mnt/data/models}"

HF="hf download"
HF_QUIET="hf download --quiet"

# ─── Helpers ────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# Parse YAML with Python (more reliable than bash yq)
get_registry_field() {
    local model_key="$1" field="$2"
    python3 -c "
import yaml, sys
r = yaml.safe_load(open('$REGISTRY'))
# Search served-models first, then physical entries
sv = r.get('served-models', {})
if '$model_key' in sv:
    ref = sv['$model_key'].get('pool_ref', '')
    if ref:
        cat, entry = ref.split('/', 1)
        m = r.get(cat, {}).get(entry, {})
        print(m.get('$field', ''))
    else:
        print(sv['$model_key'].get('$field', ''))
else:
    # Search physical entries
    for cat in r:
        if isinstance(r[cat], dict) and any(isinstance(v, dict) for v in r[cat].values()):
            if '$model_key' in r[cat]:
                print(r[cat]['$model_key'].get('$field', ''))
                break
"
}

# ─── Section definitions ────────────────────────────────────────────────────
# Each section lists its served-models keys. The actual download info comes
# from model_registry.yaml via get_registry_field.

SECTIONS_AVAILABLE="core audio image video llm embeddings special-ops"

declare -A SECTION_MODELS
SECTION_MODELS[c]="core"
SECTION_MODELS[a]="audio"

# Core: essential infrastructure models
CORE_MODELS=""

# Audio: audio/sound generation models
# Note: VibeVoice-7B TTS removed — superseded by OpenMOSS for all TTS.
# Note: Kokoro is now served via sherpa-onnx (kokoro-multi-lang-v1_0) —
# download via --kokoro-sherpa special op, not the hf:// snapshot.
# Note: qwen3-tts removed — superseded by MOSS VoiceGenerator for
# instruction-following + multilingual TTS. Kokoro (CPU) and MOSS (GPU)
# cover the full TTS surface.
# Note: tangoflux removed — superseded by MOSS SoundEffect-v2 (Tier A)
# for all sound/audio-effect generation.
AUDIO_MODELS="ace-step ace-step-turbo"

# Image: image generation models
IMAGE_MODELS="qwen-image-edit z-image z-image-base ideogram4 see-through kimodo comfyui"

# Video: video generation models  
VIDEO_MODELS="wan-vace wan-t2v wan-i2v cosmos ltx-video"

# LLM: language models
LLM_MODELS="llama llama-bee qwen3.6-27b-q5_k_s"

# Embeddings: embedding models
EMBEDDINGS_MODELS="jina-v5-nano-clustering-q5_k_m jina-v5-nano-retrieval-iq4_nl"

# ─── list-models: Show all models grouped by section ─────────────────────────
list_models() {
    echo "═══ Model Inventory (from model_registry.yaml) ═══"
    echo ""
    for section in core audio image video llm embeddings; do
        local var="${section^^}_MODELS"
        local models="${!var}"
        echo "  [$section]"
        for m in $models; do
            local desc=$(get_registry_field "$m" "description" 2>/dev/null | head -c 60)
            local path=$(get_registry_field "$m" "path" 2>/dev/null)
            echo "    $m"
            echo "      $desc"
            echo "      path: $path"
        done
        echo ""
    done
    echo "  [special-ops]"
    echo "    fp8-qwen          Build Qwen-Image-Edit FP8 weight-only"
    echo "    fp8-zimage        Build Z-Image Turbo/Base FP8"
    echo "    fp8-vace          Build Wan VACE 14B FP8"
    echo "    moss-gguf         Download MOSS GGUF models + ONNX tokenizer"
    echo "    ace-xl            Download ACE-Step XL GGUF variants"
    echo "    kokoro-sherpa     Download Kokoro sherpa-onnx (kokoro-multi-lang-v1_0)"
    echo ""
}

# ─── list-sections ───────────────────────────────────────────────────────────
list_sections() {
    echo "Available sections:"
    for s in core audio image video llm embeddings; do
        local var="${s^^}_MODELS"
        local models="${!var}"
        local count=$(echo "$models" | wc -w)
        echo "  $s  ($count models)"
    done
    echo "  special-ops  (FP8 builds, MOSS GGUF, ACE-Step XL)"
    echo ""
    echo "Special ops flags:"
    echo "  --fp8-qwen   --fp8-zimage   --fp8-vace   --moss-gguf   --ace-xl   --kokoro-sherpa"
}

# ─── Download a model from registry ─────────────────────────────────────────
download_model() {
    local model_key="$1"
    local source=$(get_registry_field "$model_key" "source" 2>/dev/null)
    local path=$(get_registry_field "$model_key" "path" 2>/dev/null)
    local download=$(get_registry_field "$model_key" "download" 2>/dev/null)

    if [[ -z "$source" || "$source" == "None" ]]; then
        warn "No source for $model_key — skipping"
        return 0
    fi

    local target="$MODELS_ROOT/$path"

    info "Downloading $model_key ..."
    info "  source: $source"
    info "  target: $target"

    if [[ -d "$target" ]] || [[ -f "$target" ]]; then
        info "  already exists at $target — skipping"
        return 0
    fi

    # Parse source URL
    case "$source" in
        hf://*)
            local repo_path="${source#hf://}"
            if [[ "$repo_path" == */*/* ]]; then
                # hf://org/repo/file — single file download
                local repo="${repo_path%/*}"
                local file="${repo_path##*/}"
                local repo_org="${repo%/*}"
                local repo_name="${repo#*/}"
                if [[ "$download" == "snapshot" ]]; then
                    $HF_QUIET "$repo" --local-dir "$target" 2>&1 | tail -1
                else
                    mkdir -p "$(dirname "$target")"
                    $HF_QUIET "$repo" --include "$file" --local-dir "$(dirname "$target")" 2>&1 | tail -1
                    # Move file to expected path
                    local downloaded="$MODELS_ROOT/$(dirname "$path")/$file"
                    if [[ -f "$downloaded" ]]; then
                        info "  downloaded to $downloaded"
                    fi
                fi
            else
                # hf://org/repo — full repo snapshot
                $HF_QUIET "$repo_path" --local-dir "$target" 2>&1 | tail -1
            fi
            ;;
        civitai://*)
            local id="${source#civitai://}"
            warn "CivitAI download requires manual intervention (model ID: $id)"
            return 0
            ;;
        modelscope://*)
            local ms_path="${source#modelscope://}"
            warn "ModelScope download not yet implemented for $ms_path"
            return 0
            ;;
        local|manual|skip)
            info "  download type: $download — no action needed"
            return 0
            ;;
        *)
            warn "Unknown source format: $source"
            return 0
            ;;
    esac

    info "  done"
}

# ─── Section downloaders ─────────────────────────────────────────────────────
download_section() {
    local section="$1"
    local var="${section^^}_MODELS"
    local models="${!var}"

    step "Downloading section: $section"
    for m in $models; do
        download_model "$m"
    done
}

download_all() {
    for section in core audio image video llm embeddings; do
        download_section "$section"
    done
}

# ─── Special operations ──────────────────────────────────────────────────────

build_fp8_qwen() {
    step "Building Qwen-Image-Edit FP8 weight-only"
    local script="$PROJECT_ROOT/scripts/prepare_qwen_img_edit_fp8.py"
    if [[ -f "$script" ]]; then
        info "Running: python3 $script"
        python3 "$script"
        info "Qwen-Image-Edit FP8 build complete"
    else
        error "Script not found: $script"
        return 1
    fi
}

build_fp8_zimage() {
    step "Building Z-Image Turbo/Base FP8"
    warn "Z-Image FP8 build script not yet created — manual conversion required"
    info "  Model dir: /mnt/data/models/native/z-image-turbo-fp8/"
    info "  Source: hf://Comfy-Org/z_image_turbo"
    info "  Run: python3 scripts/prepare_z_image_fp8.py"
}

build_fp8_vace() {
    step "Building Wan VACE 14B FP8"
    local script="$PROJECT_ROOT/scripts/convert_vace_to_fp8.py"
    if [[ -f "$script" ]]; then
        info "Running: python3 $script"
        python3 "$script"
        info "Wan VACE FP8 build complete"
    else
        error "Script not found: $script"
        return 1
    fi
}

download_moss_gguf() {
    step "Downloading MOSS GGUF models + ONNX tokenizer"
    local gguf_dir="$MODELS_ROOT/audio/moss-tts-gguf"
    local onnx_dir="$MODELS_ROOT/audio/moss-audio-tokenizer-onnx"

    mkdir -p "$gguf_dir" "$onnx_dir"

    # GGUF models — download Q4_K_M as default (best quality/size tradeoff)
    info "Downloading MOSS-TTS Q4_K_M GGUF ..."
    $HF_QUIET "OpenMOSS-Team/MOSS-TTS-GGUF" \
        --include "MOSS_TTS_Q4_K_M.gguf" \
        --local-dir "$gguf_dir" 2>&1 | tail -1

    info "Downloading MOSS-Audio-Tokenizer ONNX ..."
    $HF "OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX" \
        --local-dir "$onnx_dir" 2>&1 | tail -1

    info "MOSS GGUF download complete"
    info "  GGUF:     $gguf_dir/MOSS_TTS_Q4_K_M.gguf"
    info "  Tokenizer: $onnx_dir/"
}

download_ace_xl() {
    step "Downloading ACE-Step XL GGUF variants"

    # XL Turbo (8-step distilled)
    info "Downloading ACE-Step XL Turbo ..."
    $HF_QUIET "Serveurperso/ACE-Step-1.5-GGUF" \
        --include "acestep-v15-xl-turbo-Q8_0.gguf" \
        --local-dir "$MODELS_ROOT/audio/acestep-cpp" 2>&1 | tail -1

    # XL SFT (50-step)
    info "Downloading ACE-Step XL SFT ..."
    $HF_QUIET "Serveurperso/ACE-Step-1.5-GGUF" \
        --include "acestep-v15-xl-sft-Q8_0.gguf" \
        --local-dir "$MODELS_ROOT/audio/acestep-cpp" 2>&1 | tail -1

    # XL Base
    info "Downloading ACE-Step XL Base ..."
    $HF_QUIET "Serveurperso/ACE-Step-1.5-GGUF" \
        --include "acestep-v15-xl-base-Q8_0.gguf" \
        --local-dir "$MODELS_ROOT/audio/acestep-cpp" 2>&1 | tail -1

    # LM 4B
    info "Downloading ACE-Step LM 4B Q8_0 ..."
    $HF_QUIET "Serveurperso/ACE-Step-1.5-GGUF" \
        --include "acestep-5Hz-lm-4B-Q8_0.gguf" \
        --local-dir "$MODELS_ROOT/audio/acestep-cpp" 2>&1 | tail -1

    info "ACE-Step XL download complete"
    ls -lh "$MODELS_ROOT/audio/acestep-cpp/"*xl* "$MODELS_ROOT/audio/acestep-cpp/"*lm-4B* 2>/dev/null
}

download_kokoro_sherpa() {
    step "Downloading Kokoro TTS (sherpa-onnx, kokoro-multi-lang-v1_0)"
    local target="$MODELS_ROOT/tts/kokoro-sherpa"
    mkdir -p "$target"

    if [[ -f "$target/model.onnx" ]]; then
        info "  already exists at $target — skipping"
        return 0
    fi

    info "  downloading 344 MB tarball from k2-fsa GitHub releases ..."
    local url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
    local tmp=$(mktemp -d)
    wget -q --show-progress "$url" -O "$tmp/kokoro.tar.bz2"
    tar xf "$tmp/kokoro.tar.bz2" -C "$target" --strip-components=1
    rm -rf "$tmp"

    info "Kokoro sherpa-onnx download complete"
    info "  Location: $target"
    info "  Contents:"
    ls -lh "$target"/*.onnx "$target"/voices.bin "$target"/tokens.txt 2>/dev/null | sed 's/^/    /'
}

# ─── Main ────────────────────────────────────────────────────────────────────

main() {
    local section=""
    local dry_run=false

    PY_DOWNLOADER="$SCRIPT_DIR/download_all.py"

    # No args → delegate to universal Python downloader
    if [[ $# -eq 0 ]]; then
        if [[ -f "$PY_DOWNLOADER" ]]; then
            exec python3 "$PY_DOWNLOADER"
        else
            download_all
        fi
        exit $?
    fi

    # Parse args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --section|-s)
                section="$2"; shift 2 ;;
            --list-sections)
                list_sections; exit 0 ;;
            --list-models)
                list_models; exit 0 ;;
            --dry-run)
                dry_run=true; shift ;;
            --all|--auto)
                if [[ -f "$PY_DOWNLOADER" ]]; then
                    exec python3 "$PY_DOWNLOADER" ${dry_run:+--dry-run}
                else
                    download_all
                fi
                exit $? ;;
            --list-missing)
                exec python3 "$PY_DOWNLOADER" --list-missing
                exit $? ;;
            --list-manual)
                exec python3 "$PY_DOWNLOADER" --list-manual
                exit $? ;;
            --fp8-qwen)
                if $dry_run; then info "WOULD BUILD: Qwen-Image-Edit FP8"; else build_fp8_qwen; fi
                exit $? ;;
            --fp8-zimage)
                if $dry_run; then info "WOULD BUILD: Z-Image FP8"; else build_fp8_zimage; fi
                exit $? ;;
            --fp8-vace)
                if $dry_run; then info "WOULD BUILD: Wan VACE FP8"; else build_fp8_vace; fi
                exit $? ;;
            --moss-gguf)
                if $dry_run; then info "WOULD DOWNLOAD: MOSS GGUF"; else download_moss_gguf; fi
                exit $? ;;
            --ace-xl)
                if $dry_run; then info "WOULD DOWNLOAD: ACE-Step XL"; else download_ace_xl; fi
                exit $? ;;
            --kokoro-sherpa)
                if $dry_run; then info "WOULD DOWNLOAD: Kokoro sherpa-onnx"; else download_kokoro_sherpa; fi
                exit $? ;;
            --help|-h)
                echo "Usage: $0 [--section <name>] [--dry-run] [--all|--list-missing|--list-manual] [--fp8-qwen|--fp8-zimage|--fp8-vace|--moss-gguf|--ace-xl|--kokoro-sherpa]"
                echo ""
                list_sections
                exit 0 ;;
            *)
                error "Unknown option: $1"
                echo "Usage: $0 [--section <name>] [--dry-run] [special-op-flag]"
                exit 1 ;;
        esac
    done

    # Verify registry exists
    if [[ ! -f "$REGISTRY" ]]; then
        error "Registry not found: $REGISTRY"
        error "Run from project root or set REGISTRY path"
        exit 1
    fi

    if $dry_run; then
        info "DRY RUN — no downloads will be executed"
        info "Registry: $REGISTRY"
        info "Models root: $MODELS_ROOT"
        echo ""
    fi

    if [[ -n "$section" ]]; then
        case "$section" in
            core|audio|image|video|llm|embeddings)
                if $dry_run; then
                    local var="${section^^}_MODELS"
                    local models="${!var}"
                    info "WOULD DOWNLOAD section '$section': $models"
                else
                    download_section "$section"
                fi
                ;;
            special-ops|special_ops)
                info "Special operations:"
                info "  --fp8-qwen          Build Qwen-Image-Edit FP8"
                info "  --fp8-zimage        Build Z-Image FP8"
                info "  --fp8-vace          Build Wan VACE FP8"
                info "  --moss-gguf         Download MOSS GGUF"
                info "  --ace-xl            Download ACE-Step XL"
                info "  --kokoro-sherpa     Download Kokoro sherpa-onnx"
                echo ""
                info "Run with one of the flags above, e.g.:"
                info "  $0 --moss-gguf"
                info "  $0 --ace-xl"
                info "  $0 --fp8-qwen"
                info "  $0 --kokoro-sherpa"
                ;;
            *)
                error "Unknown section: $section"
                list_sections
                exit 1 ;;
        esac
    else
        # No section specified — download all
        if $dry_run; then
            info "WOULD DOWNLOAD: all sections"
            for s in core audio image video llm embeddings; do
                local var="${s^^}_MODELS"
                local models="${!var}"
                info "  [$s] $models"
            done
        else
            download_all
        fi
    fi

    info "Done."
}

main "$@"
