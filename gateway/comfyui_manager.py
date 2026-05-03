"""ComfyUI Extension Manager — Ray-native custom node lifecycle.

Reads config/comfyui_extensions.yaml and clones/updates extensions
directly into ComfyUI's custom_nodes directory, then installs any
required pip packages in the ComfyUI venv.

Supports ref pinning (for repos where main is broken, e.g. LTXVideo at 4c5add5)
and per-extension pip dependencies (opencv, gguf, ltx-video, etc.).

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

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "comfyui_extensions.yaml"


@ray.remote(num_gpus=0)
class ComfyUIExtensionManager:
    """Syncs ComfyUI custom_nodes from config/comfyui_extensions.yaml.

    Usage:
        manager = ray.get_actor("comfyui_ext_manager")
        ray.get(manager.sync.remote())
        ray.get(manager.status.remote())
    """

    def __init__(self):
        self._config_path = _CONFIG_PATH
        self._last_sync: dict = {}

    def sync(self) -> dict:
        """Clone/update all extensions declared in the config, then install pip deps."""
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

        results = {"synced": [], "updated": [], "failed": [], "skipped": [], "pinned": []}

        for ext in extensions:
            name = ext["name"]
            url = ext["url"]
            ref = ext.get("ref")
            target = custom_nodes / name

            try:
                if not target.exists():
                    logger.info("Cloning %s...", name)
                    clone_cmd = ["git", "clone"]
                    if not ref:
                        clone_cmd.append("--depth=1")
                    clone_cmd.extend([url, str(target)])
                    subprocess.run(
                        clone_cmd,
                        capture_output=True, text=True, timeout=120, check=True,
                    )
                    if ref:
                        logger.info("Pinning %s to %s...", name, ref[:8])
                        subprocess.run(
                            ["git", "checkout", ref],
                            cwd=str(target), capture_output=True, text=True, timeout=30, check=True,
                        )
                        results["pinned"].append({"name": name, "ref": ref})
                    else:
                        results["synced"].append(name)
                elif (target / ".git").exists():
                    if ref:
                        # Pinned: fetch + checkout exact ref, skip pull
                        current = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=str(target), capture_output=True, text=True,
                        ).stdout.strip()[:8]
                        if current != ref[:8]:
                            subprocess.run(
                                ["git", "fetch", "origin"],
                                cwd=str(target), capture_output=True, text=True, timeout=60,
                            )
                            subprocess.run(
                                ["git", "checkout", "-f", ref],
                                cwd=str(target), capture_output=True, text=True, timeout=30, check=True,
                            )
                            subprocess.run(
                                ["git", "clean", "-fd"],
                                cwd=str(target), capture_output=True, text=True, timeout=30,
                            )
                            results["pinned"].append({"name": name, "ref": ref})
                        else:
                            results["pinned"].append({"name": name, "ref": ref, "note": "already at pinned ref"})
                    else:
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

        # Install pip dependencies in ComfyUI's venv
        self._install_pip_deps(comfyui_dir, extensions)

        total = len(results["synced"]) + len(results["updated"]) + len(results["pinned"])
        logger.info("Extension sync: %d synced/updated/pinned, %d failed, %d skipped",
                    total, len(results["failed"]), len(results["skipped"]))
        self._last_sync = results
        return results

    def _install_pip_deps(self, comfyui_dir: Path, extensions: list) -> None:
        """Install pip dependencies declared per extension into ComfyUI venv."""
        venv_python = comfyui_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():
            logger.warning("ComfyUI venv not found at %s — skipping pip deps", venv_python)
            return

        all_deps = []
        for ext in extensions:
            for pkg in ext.get("pip", []):
                if pkg not in all_deps:
                    all_deps.append(pkg)

        if not all_deps:
            return

        for pkg in all_deps:
            import_name = pkg.replace("-", "_").translate(str.maketrans("", "", "[]"))
            result = subprocess.run(
                [str(venv_python), "-c", f"import {import_name}"],
                capture_output=True,
            )
            if result.returncode != 0:
                logger.info("Installing %s in ComfyUI venv...", pkg)
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install", pkg],
                    check=False, capture_output=True,
                )

        # ltx-video pins transformers<5. Re-upgrade after install.
        if "ltx-video" in all_deps:
            result = subprocess.run(
                [str(venv_python), "-c", "import transformers; print(transformers.__version__)"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip().startswith("4."):
                logger.info("Re-upgrading transformers to >=5 (downgraded by ltx-video)")
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install", "transformers>=5"],
                    check=False, capture_output=True,
                )

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
