#!/usr/bin/env bash
# Start MCP servers as persistent background processes.
#
# Usage:
#   bash scripts/start_mcp.sh          # start both
#   bash scripts/start_mcp.sh stop     # stop both
#   bash scripts/start_mcp.sh status   # check status
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/tmp/mcp-servers"
mkdir -p "$LOG_DIR"

# Read config from local.yaml via python
read_config() {
    local key="$1"
    local default="${2:-}"
    "$PROJECT_ROOT/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from registry.config import Config
val = Config().get('$key', $default)
print(val)
"
}

read_path() {
    local key="$1"
    local default="${2:-}"
    "$PROJECT_ROOT/.venv/bin/python" -c "
from pathlib import Path
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from registry.config import Config
val = Config().get('$key', '$default')
p = Path(val)
if not p.is_absolute():
    p = Path(Config().project_root) / p
print(p)
"
}

WEB_PORT=$(read_config services.mcp.local_web.port 8327)
WEB_VENV=$(read_path services.mcp.local_web.venv_python)
WEB_DIR=$(read_path services.mcp.local_web.working_dir)
MEDIA_PORT=$(read_config services.mcp.media.port 8101)
MEDIA_VENV=$(read_path services.mcp.media.venv_python)
MEDIA_DIR=$(read_path services.mcp.media.working_dir)

is_listening() {
    ss -tlnp 2>/dev/null | grep -q ":${1} " || lsof -i ":$1" >/dev/null 2>&1
}

start_web() {
    if is_listening "$WEB_PORT"; then
        echo "local-web-mcp already running on port $WEB_PORT"
        return
    fi
    echo "Starting local-web-mcp on port $WEB_PORT..."
    (cd "$WEB_DIR" && nohup "$WEB_VENV" -m uvicorn src.mcp_sse:app \
        --host 0.0.0.0 --port "$WEB_PORT" \
        > "$LOG_DIR/local-web-mcp.log" 2>&1 &)
    echo "  PID: $!"
}

start_media() {
    if is_listening "$MEDIA_PORT"; then
        echo "media-analysis-mcp already running on port $MEDIA_PORT"
        return
    fi
    echo "Starting media-analysis-mcp on port $MEDIA_PORT..."
    (cd "$MEDIA_DIR" && nohup "$MEDIA_VENV" -m uvicorn src.server:app \
        --host 0.0.0.0 --port "$MEDIA_PORT" \
        > "$LOG_DIR/media-analysis-mcp.log" 2>&1 &)
    echo "  PID: $!"
}

stop_all() {
    echo "Stopping MCP servers..."
    for port in "$WEB_PORT" "$MEDIA_PORT"; do
        pids=$(lsof -ti ":$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                kill "$pid" 2>/dev/null && echo "  Killed PID $pid (port $port)" || true
            done
        else
            echo "  Port $port: not running"
        fi
    done
}

show_status() {
    for name_port in "local-web-mcp:$WEB_PORT" "media-analysis-mcp:$MEDIA_PORT"; do
        name="${name_port%%:*}"
        port="${name_port##*:}"
        if is_listening "$port"; then
            echo "  $name: RUNNING (port $port)"
        else
            echo "  $name: STOPPED"
        fi
    done
}

cd "$PROJECT_ROOT"

case "${1:-start}" in
    start)   start_web; start_media ;;
    stop)    stop_all ;;
    status)  show_status ;;
    restart) stop_all; sleep 1; start_web; start_media ;;
    *)       echo "Usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
