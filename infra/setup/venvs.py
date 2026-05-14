"""Tech Noir Ray — Bare-metal tool venv setup.

Creates Python venvs for tools that run as bare-metal CLIToolMixin subprocesses.
Tools with compiled CUDA extensions run in Docker containers (see infra/docker/).

Usage:
    python -m infra.setup           # Set up all bare-metal tools
    python -m infra.setup ace-step  # Set up specific tool
    python -m infra.setup docker    # Build Docker worker images
    python -m infra.setup llama     # Build llama.cpp only

Docker-based tools (TRELLIS, AniGen, VibeVoice) are built via Dockerfiles,
not managed here. Run 'python -m infra.setup docker' to build them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
RAY_ROOT = INFRA_DIR.parent.parent  # infra/setup/ -> infra/ -> ray/
REPOS_DIR = RAY_ROOT / "infra" / "repos"

# CUDA 12.x toolkit for building extensions (needed for cu124/cu121 PyTorch)
CUDA_12_HOME = os.environ.get("CUDA_12_HOME", "/usr/local/cuda-12.8")


def _uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    # Fallback: check common locations
    for p in [Path.home() / ".local" / "bin" / "uv", Path("/usr/local/bin/uv")]:
        if p.is_file():
            return str(p)
    return "uv"


def _log(msg: str) -> None:
    print(f"\033[0;32m[venv]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[venv]\033[0m {msg}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command. Raises on failure with stderr shown."""
    _log(f"  $ {' '.join(str(c) for c in cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        if e.stderr:
            _warn(f"  STDERR: {e.stderr[-500:]}")
        raise


def _uv_install(venv_py: str | Path, *specs: str, **kwargs) -> None:
    """Install packages into a venv via uv pip install."""
    uv = _uv()
    cmd = [uv, "pip", "install", "--python", str(venv_py)] + list(specs)
    _run(cmd, **kwargs)


def _uv_install_from_git(venv_py: str | Path, url: str, **kwargs) -> None:
    """Install a package from a git URL."""
    _uv_install(venv_py, f"git+{url}", **kwargs)


def _can_import(venv_py: Path, module: str) -> bool:
    """Check if a module can be imported in the venv."""
    if not venv_py.is_file():
        return False
    try:
        result = subprocess.run(
            [str(venv_py), "-c", f"import {module}"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _venv_version(venv_py: Path) -> str:
    """Get Python version string from a venv."""
    try:
        result = subprocess.run(
            [str(venv_py), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "?"


# ─── ACE-Step 1.5 — text-to-music generation ─────────────────────────────

def setup_ace_step() -> bool:
    dir_ = REPOS_DIR / "ACE-Step-1.5"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "acestep"):
        _log(f"ACE-Step venv OK ({_venv_version(venv_py)})")
        return True

    if not (dir_ / "pyproject.toml").is_file():
        _warn("ACE-Step not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up ACE-Step venv (uv sync)...")
    uv = _uv()
    if not venv_py.is_file():
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
    _run(
        [uv, "sync", "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
        cwd=str(dir_),
    )
    _log("ACE-Step venv ready")
    return True


# ─── See-Through — anime character layer decomposition ────────────────────

def setup_see_through() -> bool:
    dir_ = REPOS_DIR / "see-through"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "torch") and _can_import(venv_py, "diffusers"):
        _log(f"see-through venv OK ({_venv_version(venv_py)})")
        return True

    if not (dir_ / "requirements.txt").is_file():
        _warn("see-through not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up see-through venv...")
    uv = _uv()
    if not venv_py.is_file():
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
    _uv_install(
        venv_py,
        "torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    )
    _uv_install(venv_py, "-r", str(dir_ / "requirements.txt"))
    _log("see-through venv ready")
    return True


# ─── Qwen Image Expert (Qwen3-TTS LoRA Trainer) ──────────────────────────

def setup_qwen_img() -> bool:
    dir_ = REPOS_DIR / "qwen_img_expert"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "torch"):
        _log(f"qwen_img_expert venv OK ({_venv_version(venv_py)})")
        return True

    if not (dir_ / "pyproject.toml").is_file():
        _warn("qwen_img_expert not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up qwen_img_expert venv (uv sync)...")
    uv = _uv()
    if not venv_py.is_file():
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
    _run([uv, "sync"], cwd=str(dir_))
    _log("qwen_img_expert venv ready")
    return True


# ─── ComfyUI ──────────────────────────────────────────────────────────────

def setup_comfyui() -> bool:
    dir_ = REPOS_DIR / "ComfyUI"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "torch"):
        _log(f"ComfyUI venv OK ({_venv_version(venv_py)})")
        return True

    if not dir_.is_dir():
        _warn("ComfyUI not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up ComfyUI venv...")
    uv = _uv()
    if not venv_py.is_file():
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
    _uv_install(
        venv_py,
        "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    )
    if (dir_ / "requirements.txt").is_file():
        _uv_install(venv_py, "-r", str(dir_ / "requirements.txt"))

    _log("ComfyUI venv ready")
    return True


# ─── llama.cpp — server build ────────────────────────────────────────────

def setup_llama() -> bool:
    dir_ = REPOS_DIR / "llama.cpp"

    if (dir_ / "build" / "bin" / "llama-server").is_file():
        _log("llama.cpp already built")
        return True
    if not (dir_ / "CMakeLists.txt").is_file():
        _warn("llama.cpp not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Building llama.cpp...")
    nproc = os.cpu_count() or 4
    _run(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF"],
        cwd=str(dir_),
    )
    _run(
        ["cmake", "--build", "build", "--config", "Release", f"-j{nproc}"],
        cwd=str(dir_),
    )
    _log("llama.cpp build ready")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────

TOOLS = {
    "ace-step": setup_ace_step,
    "see-through": setup_see_through,
    "qwen": setup_qwen_img,
    "comfyui": setup_comfyui,
    "llama": setup_llama,
}


def setup_docker_workers():
    """Build Docker images for CUDA-heavy tools (TRELLIS, AniGen, VibeVoice)."""
    compose_file = RAY_ROOT / "infra" / "docker" / "compose.workers.yaml"
    if not compose_file.exists():
        _warn(f"Docker workers compose file not found: {compose_file}")
        return

    _log("Building Docker worker images...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "build"],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        _warn(f"Docker build failed: {result.stderr[-500:]}")
    else:
        _log("Docker worker images built successfully.")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "docker":
        setup_docker_workers()
        return

    if target == "all":
        _log("Setting up bare-metal tool venvs...")
        for name, fn in TOOLS.items():
            print()
            try:
                fn()
            except Exception as e:
                _warn(f"  {name} setup failed: {e}")
        print()
        _log("Bare-metal tool venvs + llama.cpp build complete.")
        _log("Run 'python -m infra.setup docker' to build CUDA worker images.")
        return

    fn = TOOLS.get(target)
    if fn:
        fn()
    else:
        print(f"Usage: python -m infra.setup [{'|'.join(TOOLS)}|all|docker]")
        sys.exit(1)


if __name__ == "__main__":
    main()
