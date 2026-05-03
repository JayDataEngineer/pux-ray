"""ComfyUI Extension Manager — Ray-native custom node lifecycle.

Replaces setup_comfyui.sh with a @ray.remote actor that reads
config/comfyui_extensions.yaml and clones/updates extensions
directly into ComfyUI's custom_nodes directory.

Deployed as a named detached actor alongside GPUScheduler.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import ray
import yaml

from registry.config import Config

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "comfyui_extensions.yaml"


@ray.remote(num_gpus=0)
class ComfyUIExtensionManager:
    """Syncs ComfyUI custom_nodes from config/comfyui_extensions.yaml.

    Usage:
        manager = ray.get_actor("comfyui_ext_manager")
        await manager.sync.remote()
        await manager.status.remote()
    """

    def __init__(self):
        self._config_path = _CONFIG_PATH
        self._last_sync: dict = {}

    def sync(self) -> dict:
        """Clone/update all extensions declared in the config. Returns summary."""
        if not self._config_path.exists():
            return {"error": f"Config not found: {self._config_path}"}

        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        extensions = config.get("extensions", [])
        if not extensions:
            return {"error": "No extensions defined in config"}

        config_obj = Config()
        comfyui_dir = Path(config_obj.get(
            "services.comfyui.working_dir",
            str(config_obj.project_root.parent / "img" / "comfyui"),
        ))
        custom_nodes = comfyui_dir / "custom_nodes"
        custom_nodes.mkdir(parents=True, exist_ok=True)

        results = {"synced": [], "updated": [], "failed": [], "skipped": []}

        for ext in extensions:
            name = ext["name"]
            url = ext["url"]
            target = custom_nodes / name

            try:
                if not target.exists():
                    logger.info("Cloning %s...", name)
                    subprocess.run(
                        ["git", "clone", "--depth", "1", url, str(target)],
                        capture_output=True, text=True, timeout=120, check=True,
                    )
                    results["synced"].append(name)
                elif (target / ".git").exists():
                    logger.info("Updating %s...", name)
                    result = subprocess.run(
                        ["git", "-C", str(target), "pull", "--ff-only"],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        results["updated"].append(name)
                    else:
                        results["failed"].append({"name": name, "error": result.stderr.strip()})
                else:
                    results["skipped"].append(name)
            except subprocess.TimeoutExpired:
                results["failed"].append({"name": name, "error": "timeout"})
            except subprocess.CalledProcessError as e:
                results["failed"].append({"name": name, "error": e.stderr.strip()[:200]})

        total = len(results["synced"]) + len(results["updated"])
        logger.info("Extension sync: %d synced/updated, %d failed, %d skipped",
                    total, len(results["failed"]), len(results["skipped"]))
        self._last_sync = results
        return results

    def status(self) -> dict:
        """Return current state of all managed extensions."""
        if not self._config_path.exists():
            return {"error": "Config not found"}

        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        config_obj = Config()
        comfyui_dir = Path(config_obj.get(
            "services.comfyui.working_dir",
            str(config_obj.project_root.parent / "img" / "comfyui"),
        ))
        custom_nodes = comfyui_dir / "custom_nodes"

        extensions = []
        for ext in config.get("extensions", []):
            name = ext["name"]
            target = custom_nodes / name
            state = "missing"
            if target.exists():
                if (target / ".git").exists():
                    state = "managed"
                else:
                    state = "manual"
            extensions.append({
                "name": name,
                "state": state,
                "required": ext.get("required", False),
                "desc": ext.get("desc", ""),
            })

        return {
            "total": len(extensions),
            "managed": sum(1 for e in extensions if e["state"] == "managed"),
            "missing": sum(1 for e in extensions if e["state"] == "missing"),
            "manual": sum(1 for e in extensions if e["state"] == "manual"),
            "extensions": extensions,
            "custom_nodes_dir": str(custom_nodes),
        }

    def restart_comfyui(self) -> dict:
        """Restart ComfyUI via the Serve deployment handle to pick up new extensions."""
        try:
            from ray import serve
            handle = serve.get_deployment_handle("comfyui", "comfyui")
            ray.get(handle.stop_comfyui.remote())
            ray.get(handle.start_comfyui.remote())
            return {"status": "restarted"}
        except Exception as e:
            return {"error": str(e)}
