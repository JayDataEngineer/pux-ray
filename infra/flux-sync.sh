#!/usr/bin/env bash
set -euo pipefail

# Tech Noir Flux Sync — ensures all Flux Kustomizations are reconciling.
#
# Primary source: GitRepository (GitHub — push to master triggers sync)
# Fallback:       OCI artifact (local registry — for air-gapped / dev)
#
# Usage:
#   flux-sync.sh              Push OCI artifact + reconcile all layers
#   flux-sync.sh push         Push OCI artifact only
#   flux-sync.sh reconcile    Reconcile all Kustomizations
#   flux-sync.sh watch        Watch for file changes (dev mode)
#   flux-sync.sh switch       Switch from OCI to GitRepository source

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FLUX_DIR="$PROJECT_DIR/infra/flux"
LOCKFILE="/tmp/flux-sync.lock"
LOGFILE="$PROJECT_DIR/logs/flux-sync.log"

# All Kustomization layers in dependency order
LAYERS=(
  namespaces
  infra-storage
  infra-secrets
  helm
  git
  infra-services
  ai-services
  mcp
  networking
)

mkdir -p "$(dirname "$LOGFILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# Acquire lock
exec 200>"$LOCKFILE"
flock -n 200 || {
  log "Sync already running, skipping"
  exit 0
}

push_oci_artifact() {
  if ! REGISTRY_IP=$(kubectl get svc forge-registry -n infra -o jsonpath='{.spec.clusterIP}' 2>/dev/null); then
    log "WARN: forge-registry not found — skipping OCI push"
    return 0
  fi

  ARTIFACT_URL="oci://${REGISTRY_IP}:5000/flux-manifests"
  log "Pushing OCI artifact from $FLUX_DIR ..."

  if flux push artifact "$ARTIFACT_URL:latest" \
    --path="$FLUX_DIR" \
    --source="$(git -C "$PROJECT_DIR" config --get remote.origin.url 2>/dev/null || echo 'local')" \
    --revision="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo 'unknown')" \
    --insecure-registry 2>&1 | tee -a "$LOGFILE"; then
    log "OCI artifact pushed"
  else
    log "WARN: OCI artifact push failed (non-fatal — GitRepository is primary)"
  fi
}

reconcile_kustomizations() {
  log "Reconciling Flux Kustomizations..."
  for layer in "${LAYERS[@]}"; do
    if kubectl get kustomization "$layer" -n flux-system &>/dev/null; then
      log "  Reconciling $layer..."
      flux reconcile kustomization "$layer" --with-source 2>&1 | tee -a "$LOGFILE" || true
    else
      log "  SKIP $layer (not yet created)"
    fi
  done
  log "Reconciliation complete"
}

switch_to_git_source() {
  log "Switching to GitRepository source..."
  KUSTOMIZATION_FILE="$PROJECT_DIR/infra/flux/clusters/forge/kustomization.yaml"
  if [[ ! -f "$KUSTOMIZATION_FILE" ]]; then
    log "ERROR: kustomization.yaml not found at $KUSTOMIZATION_FILE"
    return 1
  fi

  # Remove old OCI-based Kustomizations that conflict by name
  for layer in "${LAYERS[@]}"; do
    existing_source=$(kubectl get kustomization "$layer" -n flux-system -o jsonpath='{.spec.sourceRef.kind}' 2>/dev/null || echo "")
    if [[ "$existing_source" == "OCIRepository" ]]; then
      log "  Updating $layer from OCIRepository to GitRepository..."
    fi
  done

  # Apply the full kustomization.yaml (creates GitRepository + all Kustomization layers)
  kubectl apply -f "$KUSTOMIZATION_FILE"
  log "GitRepository source applied — Flux will sync from GitHub"

  # Remove OCIRepository (no longer primary)
  if kubectl get ocirepository flux-manifests -n flux-system &>/dev/null; then
    log "Removing OCIRepository flux-manifests (GitRepository is now primary)..."
    kubectl delete ocirepository flux-manifests -n flux-system || true
  fi
}

watch_loop() {
  log "Starting flux-sync watcher on $FLUX_DIR"
  while true; do
    inotifywait -r -e modify,create,delete,move \
      --exclude '\.(swp|swx|~|bak)' \
      "$FLUX_DIR" 2>/dev/null || sleep 5
    sleep 2  # Debounce
    push_oci_artifact
    reconcile_kustomizations
  done
}

case "${1:-}" in
  watch)
    watch_loop
    ;;
  push)
    push_oci_artifact
    ;;
  reconcile)
    reconcile_kustomizations
    ;;
  switch)
    switch_to_git_source
    ;;
  *)
    push_oci_artifact
    reconcile_kustomizations
    ;;
esac
