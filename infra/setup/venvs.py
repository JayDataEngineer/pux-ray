"""Tech Noir Ray — Creative Tool Venv Setup.

Creates Python venvs for creative tools and builds llama.cpp.
Idempotent — safe to re-run. Skips tools with existing working venvs.

Usage:
    python -m infra.setup           # Set up all tools
    python -m infra.setup trellis   # Set up TRELLIS only
    python -m infra.setup llama     # Build llama.cpp only
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
REPOS_DIR = INFRA_DIR / "repos"
RAY_ROOT = INFRA_DIR.parent


def _uv() -> str:
    uv = shutil.which("uv")
    return uv or "uv"


def _log(msg: str) -> None:
    print(f"\033[0;32m[venv]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[venv]\033[0m {msg}")


def _venv_ok(venv_py: Path) -> bool:
    """Check if a venv Python works and can import torch."""
    if not venv_py.is_file():
        return False
    try:
        result = subprocess.run(
            [str(venv_py), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _venv_version(venv_py: Path) -> str:
    try:
        result = subprocess.run(
            [str(venv_py), "--version"], capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ─── TRELLIS.2 — image-to-3D mesh generation ─────────────────────────────

def setup_trellis() -> bool:
    dir_ = REPOS_DIR / "TRELLIS.2"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _venv_ok(venv_py):
        _log(f"TRELLIS.2 venv OK ({_venv_version(venv_py)})")
        return True
    if not (dir_ / "setup.sh").is_file():
        _warn("TRELLIS.2 not cloned yet. Run: docker compose run --rm tools-sync")
        return False

    _log("Setting up TRELLIS.2 venv (CUDA extensions: o-voxel)...")
    subprocess.run(["bash", "setup.sh", "--new-env", "--basic", "--o-voxel"], cwd=str(dir_), check=True)
    _log("TRELLIS.2 venv ready")
    return True


# ─── AniGen — animated 3D character generation ───────────────────────────

def setup_anigen() -> bool:
    dir_ = REPOS_DIR / "AniGen"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _venv_ok(venv_py):
        _log(f"AniGen venv OK ({_venv_version(venv_py)})")
        return True
    if not (dir_ / "setup.sh").is_file():
        _warn("AniGen not cloned yet. Run: docker compose run --rm tools-sync")
        return False

    _log("Setting up AniGen venv...")
    subprocess.run(["bash", "setup.sh", "--new-env", "--all"], cwd=str(dir_), check=True)
    _log("AniGen venv ready")
    return True


# ─── ACE-Step 1.5 — text-to-music generation ─────────────────────────────

def setup_ace_step() -> bool:
    dir_ = REPOS_DIR / "ACE-Step-1.5"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _venv_ok(venv_py):
        _log(f"ACE-Step venv OK ({_venv_version(venv_py)})")
        return True
    if not (dir_ / "pyproject.toml").is_file():
        _warn("ACE-Step not cloned yet. Run: docker compose run --rm tools-sync")
        return False

    _log("Setting up ACE-Step venv (uv sync)...")
    uv = _uv()
    subprocess.run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_), check=True)
    subprocess.run(
        [uv, "sync", "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
        cwd=str(dir_), check=True,
    )
    _log("ACE-Step venv ready")
    return True


# ─── See-Through — anime character layer decomposition ────────────────────

def setup_see_through() -> bool:
    dir_ = REPOS_DIR / "see-through"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _venv_ok(venv_py):
        _log(f"see-through venv OK ({_venv_version(venv_py)})")
        return True
    if not (dir_ / "requirements.txt").is_file():
        _warn("see-through not cloned yet. Run: docker compose run --rm tools-sync")
        return False

    _log("Setting up see-through venv...")
    uv = _uv()
    subprocess.run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_), check=True)
    subprocess.run(
        [
            uv, "pip", "install",
            "torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        ],
        cwd=str(dir_), check=True,
    )
    subprocess.run(
        [uv, "pip", "install", "-r", "requirements.txt"],
        cwd=str(dir_), check=True,
    )
    _log("see-through venv ready")
    return True


# ─── Qwen Image Expert (Qwen3-TTS LoRA Trainer) ──────────────────────────

def setup_qwen_img() -> bool:
    dir_ = REPOS_DIR / "qwen_img_expert"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _venv_ok(venv_py):
        _log(f"qwen_img_expert venv OK ({_venv_version(venv_py)})")
        return True
    if not (dir_ / "pyproject.toml").is_file():
        _warn("qwen_img_expert not cloned yet. Run: docker compose run --rm tools-sync")
        return False

    _log("Setting up qwen_img_expert venv (uv sync)...")
    uv = _uv()
    subprocess.run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_), check=True)
    subprocess.run([uv, "sync"], cwd=str(dir_), check=True)
    _log("qwen_img_expert venv ready")
    return True


# ─── llama.cpp — server build ────────────────────────────────────────────

def setup_llama() -> bool:
    dir_ = REPOS_DIR / "llama.cpp"

    if (dir_ / "build" / "bin" / "llama-server").is_file():
        _log("llama.cpp already built")
        return True
    if not (dir_ / "CMakeLists.txt").is_file():
        _warn("llama.cpp not cloned yet. Run: docker compose run --rm llama-sync")
        return False

    _log("Building llama.cpp...")
    import os
    nproc = os.cpu_count() or 4
    subprocess.run(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF"],
        cwd=str(dir_), check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", f"-j{nproc}"],
        cwd=str(dir_), check=True,
    )
    _log("llama.cpp build ready")
    return True


# ─── Migrate existing venvs from old locations ───────────────────────────

def migrate_venvs() -> None:
    migrations = [
        ("/home/ubuntu/Documents/programs/TRELLIS.2", "TRELLIS.2"),
        ("/home/ubuntu/Documents/programs/AniGen", "AniGen"),
        ("/home/ubuntu/Documents/programs/vid/ACE-Step-1.5", "ACE-Step-1.5"),
        ("/home/ubuntu/Documents/programs/creative/see-through", "see-through"),
        ("/home/ubuntu/Documents/programs/creative/qwen_img_expert", "qwen_img_expert"),
    ]
    for old_path, name in migrations:
        old = Path(old_path)
        new = REPOS_DIR / name
        if (old / ".venv").is_dir() and not (new / ".venv").exists():
            _log(f"Migrating {name} venv: {old}/.venv -> {new}/.venv")
            shutil.copytree(str(old / ".venv"), str(new / ".venv"))
        elif not (old / ".venv").is_dir() and (new / ".venv").is_dir():
            _log(f"{name} venv already at new location")
    _log("Migration complete.")


# ─── Main ─────────────────────────────────────────────────────────────────

TOOLS = {
    "trellis": setup_trellis,
    "anigen": setup_anigen,
    "ace-step": setup_ace_step,
    "see-through": setup_see_through,
    "qwen": setup_qwen_img,
    "llama": setup_llama,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "migrate":
        migrate_venvs()
        return

    if target == "all":
        _log("Setting up all creative tools...")
        for name, fn in TOOLS.items():
            print()
            try:
                fn()
            except Exception as e:
                _warn(f"  {name} setup had issues: {e}")
        print()
        _log("All creative tool venvs + llama.cpp build complete.")
        return

    fn = TOOLS.get(target)
    if fn:
        fn()
    else:
        print(f"Usage: python -m infra.setup [{'|'.join(TOOLS)}|all|migrate]")
        sys.exit(1)


if __name__ == "__main__":
    main()
