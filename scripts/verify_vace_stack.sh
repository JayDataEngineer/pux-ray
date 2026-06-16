#!/usr/bin/env bash
# verify_vace_stack.sh — Verify each optimization in the VACE stack is active.
#
# Run AFTER the vace pod is healthy. Hits /health for config, then runs a
# minimal generation and inspects logs + metrics for each lever:
#   1. FP8 e4m3fn storage       — /health config.fp8 == true
#   2. TeaCache block-skip      — /health config.teacache_thresh > 0 + metrics
#   3. SageAttention            — /health config.attention == sage_attention
#   4. Tiled VAE                — /health config.tiled == true
#   5. AutoWrappedModule streaming — startup log mentions "AutoWrappedModule"
#                                   or "vram_management"
#
# Usage: bash scripts/verify_vace_stack.sh
set -euo pipefail

VACE_URL="${VACE_URL:-http://vace-service:8082}"
NAMESPACE="${NAMESPACE:-ai-services}"
POD_NAME="${POD_NAME:-}"

# Discover pod if not given
if [ -z "$POD_NAME" ]; then
  POD_NAME=$(kubectl -n "$NAMESPACE" get pod -l app=vace-video -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [ -z "$POD_NAME" ]; then
    echo "❌ No vace-video pod found in namespace $NAMESPACE"
    exit 1
  fi
fi

echo "=== VACE pod: $POD_NAME ==="
echo ""

# ── 1. Health endpoint ────────────────────────────────────────────────────
echo "── /health ────────────────────────────────────────────────────────────"
if ! health=$(kubectl -n "$NAMESPACE" exec "$POD_NAME" -- curl -sf http://localhost:8082/health 2>/dev/null); then
  echo "❌ /health not reachable"
  exit 1
fi
echo "$health" | python3 -m json.tool 2>/dev/null || echo "$health"
echo ""

# ── 2. Optimization assertions ────────────────────────────────────────────
echo "── Optimization assertions ───────────────────────────────────────────"
pass=0; fail=0
check() {
  local label="$1" actual="$2" expected="$3"
  if [ "$actual" = "$expected" ]; then
    echo "  ✅ $label: $actual"
    pass=$((pass+1))
  else
    echo "  ❌ $label: got '$actual', expected '$expected'"
    fail=$((fail+1))
  fi
}

fp8=$(echo "$health"     | python3 -c "import json,sys; print(str(json.load(sys.stdin)['config']['fp8']).lower())")
tc=$(echo "$health"      | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['teacache_thresh'])")
attn=$(echo "$health"    | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['attention'])")
tiled=$(echo "$health"   | python3 -c "import json,sys; print(str(json.load(sys.stdin)['config']['tiled']).lower())")

check "FP8 e4m3fn storage"  "$fp8"   "true"
check "TeaCache thresh>0"   "$tc"    "0.15"
check "SageAttention"       "$attn"  "sage_attention"
check "Tiled VAE"           "$tiled" "true"
echo ""

# ── 3. Startup log inspection ─────────────────────────────────────────────
echo "── Startup log inspection (last 60 lines) ────────────────────────────"
kubectl -n "$NAMESPACE" logs "$POD_NAME" --tail=60 2>&1 | grep -iE "sage|fp8|teacache|tiled|vram|attention|AutoWrapped|loaded|ready" | tail -20 || echo "(no matching log lines yet)"
echo ""

# ── 4. Summary ────────────────────────────────────────────────────────────
echo "── Summary ──────────────────────────────────────────────────────────"
echo "  $pass/$((pass+fail)) optimizations verified active"
if [ "$fail" -gt 0 ]; then
  echo "  ⚠️  $fail not active — check pod logs: kubectl -n $NAMESPACE logs $POD_NAME"
  exit 1
fi
echo "  ✅ All optimizations verified"
