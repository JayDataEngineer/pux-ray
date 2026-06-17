#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Auto GPU Eviction System
# ════════════════════════════════════════════════════════════════════════════
# Gracefully stops/restarts inference containers, evicts GPU memory, and
# reports VRAM status. Run BEFORE switching between inference services to
# prevent OOM crashes.
#
# Usage:
#   ./auto_evict_gpu.sh                     # stop all inference containers
#   ./auto_evict_gpu.sh --dry-run           # show what would be stopped
#   ./auto_evict_gpu.sh --status            # show GPU status only
#   ./auto_evict_gpu.sh --keep diarization  # stop all EXCEPT named containers
#   ./auto_evict_gpu.sh --container omni-qwen-img-edit-fp8  # stop specific
#   ./auto_evict_gpu.sh --restart omni-z-image-fp8           # stop + wait + start
#
# Auto-discovery: scans all containers with 'inference-' or 'omni-' prefix.
# SHEOF

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

# Parse arguments
DRY_RUN=false
STATUS_ONLY=false
KEEP_CONTAINERS=""
TARGET_CONTAINER=""
RESTART_CONTAINER=""
RESTART_SCRIPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --status|--status-only) STATUS_ONLY=true; shift ;;
    --keep) KEEP_CONTAINERS="$2"; shift 2 ;;
    --container) TARGET_CONTAINER="$2"; shift 2 ;;
    --restart) RESTART_CONTAINER="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── GPU Status ──────────────────────────────────────────────────────────
get_gpu_status() {
  if ! command -v nvidia-smi &>/dev/null; then
    echo "  ⚠  nvidia-smi not found"
    return
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s: %s | VRAM: %s / %s | Util: %s\n", $1, $2, $3, $4, $5}'
}

# ── Find Inference Containers ───────────────────────────────────────────
find_inference_containers() {
  docker ps --format "{{.Names}}" 2>/dev/null | grep -E '^(inference-|omni-)' || true
}

# ── Stop Container ──────────────────────────────────────────────────────
stop_container() {
  local name="$1"
  if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would stop: $name"
    return
  fi
  echo -n "  Stopping $name... "
  if docker stop "$name" >/dev/null 2>&1; then
    echo "✓"
    echo -n "  Removing $name... "
    docker rm "$name" >/dev/null 2>&1 && echo "✓" || echo "(already removed)"
  else
    echo "(not running)"
  fi
}

# ── Clear GPU Memory ────────────────────────────────────────────────────
clear_gpu_memory() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would clear GPU memory"
    return
  fi
  echo "  Waiting for GPU memory to be released..."
  # Run nvidia-smi to force reclaim
  nvidia_smi_out=$(nvidia-smi 2>/dev/null) || true
  # Try to clear via a quick Python script
  python3 -c "
import torch, gc, time
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    # Reset peak stats
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / 1024**3
    total_gb = total / 1024**3
    print(f'  ✓ GPU memory cleared. Used: {used_gb:.1f}GB / {total_gb:.1f}GB')
else:
    print('  ⚠  No CUDA GPU available')
" 2>/dev/null || {
    sleep 3
    echo "  (waited 3s for memory release)"
  }
}

# ── Main ────────────────────────────────────────────────────────────────
echo "═══ Auto GPU Eviction System ═══════════════════════════════════════"
echo ""

if [[ "$STATUS_ONLY" == true ]]; then
  echo "GPU Status:"
  get_gpu_status
  echo ""
  echo "Running inference containers:"
  containers=$(find_inference_containers)
  if [[ -z "$containers" ]]; then
    echo "  (none)"
  else
    for c in $containers; do
      echo "  • $c"
    done
  fi
  exit 0
fi

echo "GPU Status (before):"
get_gpu_status
echo ""

# Find containers to evict
if [[ -n "$TARGET_CONTAINER" ]]; then
  CONTAINERS_TO_STOP="$TARGET_CONTAINER"
elif [[ -n "$KEEP_CONTAINERS" ]]; then
  CONTAINERS_TO_STOP=""
  for c in $(find_inference_containers); do
    keep=false
    for k in $KEEP_CONTAINERS; do
      if [[ "$c" == "$k" ]]; then
        keep=true
        break
      fi
    done
    if [[ "$keep" == false ]]; then
      CONTAINERS_TO_STOP="$CONTAINERS_TO_STOP $c"
    fi
  done
else
  CONTAINERS_TO_STOP=$(find_inference_containers)
fi

if [[ -z "$CONTAINERS_TO_STOP" ]]; then
  echo "No inference containers to evict."
else
  echo "Evicting containers:"
  for c in $CONTAINERS_TO_STOP; do
    stop_container "$c"
  done
  echo ""
  clear_gpu_memory
fi

echo ""
echo "GPU Status (after):"
get_gpu_status

# ── Restart if requested ────────────────────────────────────────────────
if [[ -n "$RESTART_CONTAINER" ]]; then
  echo ""
  echo "═══ Restarting: $RESTART_CONTAINER ══════════════════════════════════"
  # Auto-discover launch script from pattern
  # Map container name → script name
  case "$RESTART_CONTAINER" in
    omni-qwen-img-edit-fp8)    RESTART_SCRIPT="scripts/run_omni_qwen_img_edit_fp8.sh" ;;
    omni-z-image-fp8)          RESTART_SCRIPT="scripts/run_omni_z_image_fp8.sh" ;;
    inference-moss)             RESTART_SCRIPT="scripts/run_moss_server.sh" ;;
    inference-diarization)      RESTART_SCRIPT="scripts/run_diarization.sh" ;;
    inference-ace-step)         RESTART_SCRIPT="scripts/run_ace_step.sh" ;;
    omni-14b-vace-fp8)          RESTART_SCRIPT="scripts/run_omni_14b.sh" ;;
    *)                          echo "  ⚠  No known restart script for $RESTART_CONTAINER"
                                echo "  Restart manually with the appropriate run script."
                                exit 1 ;;
  esac
  echo "  Running: $RESTART_SCRIPT"
  cd "$(dirname "$0")/.."
  if [[ "$DRY_RUN" == false ]]; then
    exec bash "$RESTART_SCRIPT"
  else
    echo "  [DRY RUN] Would execute: $RESTART_SCRIPT"
  fi
fi
