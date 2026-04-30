"""GPU metrics collector and dashboard API endpoints.

Background thread polls nvidia-smi every 5 seconds, maintaining a rolling
5-minute history buffer. Starlette async endpoints serve JSON to the frontend.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Optional

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU sample dataclass
# ---------------------------------------------------------------------------

NVIDIA_SMI_FIELDS = (
    "index,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.free,memory.total,"
    "temperature.gpu,power.draw,power.limit,fan.speed"
)

NA_VALUES = ("[N/A]", "[Not Supported]", "N/A", "Not Supported", "")


@dataclass
class GPUSample:
    timestamp: float
    index: int
    name: str
    gpu_util: int
    mem_util: int
    mem_used_mb: int
    mem_free_mb: int
    mem_total_mb: int
    temp_c: int
    power_w: float
    power_limit_w: float
    fan_speed: int


# ---------------------------------------------------------------------------
# Background metrics collector
# ---------------------------------------------------------------------------

class GPUMetricsCollector:
    """Daemon thread polling nvidia-smi at a fixed interval.

    Maintains a rolling deque of GPUSample objects (default 60 samples / 5 min).
    """

    def __init__(self, interval: int = 5, history_size: int = 60):
        self.interval = interval
        self._history: deque[GPUSample] = deque(maxlen=history_size)
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Synchronous first poll so data is available immediately
        sample = self._poll_nvidia_smi()
        if sample:
            self._history.append(sample)
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("GPU metrics collector started (interval=%ds)", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def history(self) -> list[GPUSample]:
        return list(self._history)

    @property
    def latest(self) -> Optional[GPUSample]:
        return self._history[-1] if self._history else None

    # -- private --

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._poll_nvidia_smi()
            if sample:
                self._history.append(sample)
            self._stop_event.wait(self.interval)

    @staticmethod
    def _safe_int(val: str, default: int = 0) -> int:
        if val in NA_VALUES:
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(val: str, default: float = 0.0) -> float:
        if val in NA_VALUES:
            return default
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return default

    def _poll_nvidia_smi(self) -> Optional[GPUSample]:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_FIELDS}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 11:
                return None
            return GPUSample(
                timestamp=time.time(),
                index=self._safe_int(parts[0]),
                name=parts[1],
                gpu_util=self._safe_int(parts[2]),
                mem_util=self._safe_int(parts[3]),
                mem_used_mb=self._safe_int(parts[4]),
                mem_free_mb=self._safe_int(parts[5]),
                mem_total_mb=self._safe_int(parts[6]),
                temp_c=self._safe_int(parts[7]),
                power_w=self._safe_float(parts[8]),
                power_limit_w=self._safe_float(parts[9]),
                fan_speed=self._safe_int(parts[10]),
            )
        except Exception as e:
            logger.debug("nvidia-smi poll failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Service status registry
# ---------------------------------------------------------------------------

KNOWN_DEPLOYMENTS = {
    "llm":             {"label": "LLM (llama.cpp)", "category": "LLM",      "gpu": True},
    "kokoro_tts":      {"label": "Kokoro TTS",      "category": "TTS",      "gpu": False},
    "espeak_tts":      {"label": "eSpeak TTS",      "category": "TTS",      "gpu": False},
    "index_tts":       {"label": "IndexTTS",        "category": "TTS",      "gpu": True},
    "qwen_tts":        {"label": "Qwen3-TTS",       "category": "TTS",      "gpu": True},
    "vibevoice":       {"label": "VibeVoice TTS",   "category": "TTS",      "gpu": True},
    "gpt_sovits":      {"label": "GPT-SoVITS",      "category": "TTS",      "gpu": True},
    "faster_whisper":  {"label": "Faster-Whisper",  "category": "ASR",      "gpu": False},
    "vibevoice_asr":   {"label": "VibeVoice ASR",   "category": "ASR",      "gpu": True},
    "qwen_asr":        {"label": "Qwen ASR",        "category": "ASR",      "gpu": True},
    "comfyui":         {"label": "ComfyUI",         "category": "Image",    "gpu": True},
    "trellis":         {"label": "TRELLIS.2",       "category": "3D",       "gpu": True},
    "anigen":          {"label": "AniGen",          "category": "3D",       "gpu": True},
    "see_through":     {"label": "See-Through",     "category": "Creative", "gpu": True},
    "ace_step":        {"label": "ACE-Step",        "category": "Music",    "gpu": True},
    "local_web_mcp":   {"label": "Local Web MCP",   "category": "MCP",      "gpu": False},
    "media_analysis_mcp": {"label": "Media Analysis MCP", "category": "MCP", "gpu": False},
}


def query_service_status() -> list[dict]:
    """Query Ray Serve for all known deployment statuses."""
    try:
        from ray import serve
        status = serve.status()
    except Exception:
        return []

    # Build flat lookup: deployment_name -> DeploymentStatus
    deployed: dict[str, object] = {}
    try:
        for app_name, app_details in status.applications.items():
            for dep_name, dep in app_details.deployments.items():
                deployed[dep_name] = dep
    except Exception:
        pass

    services = []
    for dep_name, meta in KNOWN_DEPLOYMENTS.items():
        dep = deployed.get(dep_name)
        if dep:
            running = getattr(dep, "running_replicas", 0)
            target = getattr(dep, "target_replicas", 1)
            status_val = getattr(dep, "status", None)
            status_str = status_val.value if hasattr(status_val, "value") else str(status_val)
            services.append({
                "name": dep_name,
                "label": meta["label"],
                "category": meta["category"],
                "gpu": meta["gpu"],
                "status": status_str,
                "running_replicas": running,
                "target_replicas": target,
            })
        else:
            services.append({
                "name": dep_name,
                "label": meta["label"],
                "category": meta["category"],
                "gpu": meta["gpu"],
                "status": "NOT_DEPLOYED",
                "running_replicas": 0,
                "target_replicas": 0,
            })

    return services


def query_gpu_processes() -> list[dict]:
    """Get current GPU compute processes via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        procs = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                procs.append({
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "memory_mb": int(parts[2]),
                })
        return procs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector = GPUMetricsCollector()


def start_collector() -> None:
    _collector.start()


def stop_collector() -> None:
    _collector.stop()


# ---------------------------------------------------------------------------
# Starlette endpoint functions
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


async def dashboard_page(request: Request) -> HTMLResponse:
    """GET /dashboard — serve the single-page dashboard HTML."""
    return HTMLResponse(_DASHBOARD_HTML.read_text())


async def dashboard_gpu_current(request: Request) -> JSONResponse:
    """GET /dashboard/api/gpu — current GPU snapshot + scheduler + processes."""
    sample = _collector.latest
    if sample is None:
        return JSONResponse({"error": "no data yet"}, status_code=503)

    # Query GPU scheduler state
    scheduler_state = {}
    try:
        import ray
        scheduler = ray.get_actor("gpu_scheduler")
        scheduler_state = await scheduler.status.remote()
    except Exception:
        pass

    data = asdict(sample)
    data["scheduler"] = scheduler_state
    data["processes"] = query_gpu_processes()
    return JSONResponse(data)


async def dashboard_gpu_history(request: Request) -> JSONResponse:
    """GET /dashboard/api/gpu/history — rolling 5-min samples for sparklines."""
    return JSONResponse([
        {
            "t": s.timestamp,
            "gpu_util": s.gpu_util,
            "mem_util": s.mem_util,
            "mem_used_mb": s.mem_used_mb,
            "temp_c": s.temp_c,
            "power_w": s.power_w,
        }
        for s in _collector.history
    ])


async def dashboard_services(request: Request) -> JSONResponse:
    """GET /dashboard/api/services — all Ray Serve deployment statuses."""
    return JSONResponse(query_service_status())
