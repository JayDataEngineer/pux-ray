"""Tech Noir Ray — Creative Tool Venv Setup.

Creates Python venvs for creative tools with compiled CUDA extensions.
Idempotent — safe to re-run. Skips steps that are already complete.

Usage:
    python -m infra.setup           # Set up all tools
    python -m infra.setup trellis   # Set up TRELLIS only
    python -m infra.setup llama     # Build llama.cpp only

Architecture:
    Each tool has its own venv with incompatible torch/CUDA versions:
    - TRELLIS.2:  torch 2.6.0+cu124, Python 3.12
    - AniGen:     torch 2.5.0+cu121, Python 3.12
    - ACE-Step:   torch 2.10.0+cu128, Python 3.12 (uv sync)
    - See-Through: torch 2.8.0+cu128, Python 3.12
    - VibeVoice:  torch + transformers 4.51.3 (pinned)
    - GPT-SoVITS: torch + SoVITS inference

    Compiled extensions (flash-attn, pytorch3d, cumesh, o_voxel, nvdiffrast)
    are built against the CUDA 12.x toolkit. Set CUDA_12_HOME to the path
    (default: /usr/local/cuda-12.8).

    RTX 4090 = sm_89 (set via TORCH_CUDA_ARCH_LIST).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
RAY_ROOT = INFRA_DIR.parent.parent  # infra/setup/ -> infra/ -> ray/
REPOS_DIR = RAY_ROOT / "infra" / "repos"

# CUDA 12.x toolkit for building extensions (needed for cu124/cu121 PyTorch)
CUDA_12_HOME = os.environ.get("CUDA_12_HOME", "/usr/local/cuda-12.8")
TORCH_CUDA_ARCH = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")


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
    try:
        result = subprocess.run(
            [str(venv_py), "--version"], capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _build_ext_env() -> dict[str, str]:
    """Environment for building CUDA extensions."""
    env = dict(os.environ)
    env["CUDA_HOME"] = CUDA_12_HOME
    env["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH
    env["PATH"] = str(Path.home() / ".local" / "bin") + ":" + env.get("PATH", "")
    return env


def _build_from_source(venv_py: Path, source_dir: str) -> None:
    """Build and install a CUDA extension from source dir."""
    uv = _uv()
    cmd = [uv, "pip", "install", "--python", str(venv_py), source_dir, "--no-build-isolation"]
    _run(cmd, env=_build_ext_env())


def _clone_to_tmp(url: str, name: str, branch: str | None = None) -> str:
    """Clone a repo to /tmp and return the path."""
    dest = f"/tmp/tech_noir_build/{name}"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cmd = ["git", "clone", "--recursive", url, dest]
    if branch:
        cmd = ["git", "clone", "--recursive", "--branch", branch, url, dest]
    _run(cmd)
    return dest


# ─── TRELLIS.2 — image-to-3D mesh generation ─────────────────────────────

def _trellis_ok(venv_py: Path) -> bool:
    """Check if TRELLIS venv has all compiled extensions."""
    for mod in ["torch", "PIL", "flash_attn", "cumesh", "o_voxel", "nvdiffrast.torch"]:
        if not _can_import(venv_py, mod):
            return False
    return True


def setup_trellis() -> bool:
    dir_ = REPOS_DIR / "TRELLIS.2"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _trellis_ok(venv_py):
        _log(f"TRELLIS.2 venv OK ({_venv_version(venv_py)})")
        return True

    if not dir_.is_dir():
        _warn("TRELLIS.2 not cloned. Run: python -m infra.setup.clone")
        return False

    # Step 1: Create venv if needed
    if not venv_py.is_file():
        _log("Creating TRELLIS.2 venv (Python 3.12, torch cu124)...")
        uv = _uv()
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
        _uv_install(
            venv_py,
            "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        )

    # Step 2: Basic dependencies
    _log("Installing TRELLIS.2 basic dependencies...")
    _uv_install(
        venv_py,
        "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
        "ninja", "trimesh", "transformers", "tensorboard", "pandas", "lpips",
        "zstandard", "kornia", "timm", "gradio==6.0.1",
    )
    _uv_install(venv_py, "wheel", "setuptools>=70.1")

    # Step 3: flash-attn (pre-built wheel available)
    if not _can_import(venv_py, "flash_attn"):
        _log("Installing flash-attn (pre-built wheel)...")
        _uv_install(venv_py, "flash-attn", "--no-build-isolation")

    # Step 4: CuMesh (compiled, needs CUDA 12 toolkit)
    if not _can_import(venv_py, "cumesh"):
        _log("Building CuMesh from source...")
        src = _clone_to_tmp("https://github.com/JeffreyXiang/CuMesh.git", "CuMesh")
        _build_from_source(venv_py, src)

    # Step 5: o-voxel (compiled, from TRELLIS repo)
    if not _can_import(venv_py, "o_voxel"):
        _log("Building o-voxel from source...")
        _build_from_source(venv_py, str(dir_ / "o-voxel"))

    # Step 6: nvdiffrast (compiled)
    if not _can_import(venv_py, "nvdiffrast"):
        _log("Building nvdiffrast from source...")
        src = _clone_to_tmp("https://github.com/NVlabs/nvdiffrast.git", "nvdiffrast", branch="v0.4.0")
        _build_from_source(venv_py, src)

    # Step 7: FlexGEMM (compiled)
    if not _can_import(venv_py, "flex_gemm"):
        _log("Building FlexGEMM from source...")
        src = _clone_to_tmp("https://github.com/JeffreyXiang/FlexGEMM.git", "FlexGEMM")
        _build_from_source(venv_py, src)

    # Step 8: nvdiffrec (compiled)
    if not _can_import(venv_py, "renderutils"):
        _log("Building nvdiffrec from source...")
        src = _clone_to_tmp("https://github.com/JeffreyXiang/nvdiffrec.git", "nvdiffrec", branch="renderutils")
        _build_from_source(venv_py, src)

    # Step 9: utils3d (TRELLIS dependency)
    _uv_install(venv_py, "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")

    # Verify
    if _trellis_ok(venv_py):
        _log(f"TRELLIS.2 venv ready ({_venv_version(venv_py)})")
        return True
    else:
        _warn("TRELLIS.2 venv incomplete — some extensions may have failed")
        return False


# ─── AniGen — animated 3D character generation ───────────────────────────

def _anigen_ok(venv_py: Path) -> bool:
    """Check if AniGen venv has all compiled extensions."""
    for mod in ["torch", "pytorch3d", "spconv", "flash_attn"]:
        if not _can_import(venv_py, mod):
            return False
    return True


def setup_anigen() -> bool:
    dir_ = REPOS_DIR / "AniGen"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _anigen_ok(venv_py):
        _log(f"AniGen venv OK ({_venv_version(venv_py)})")
        return True

    if not dir_.is_dir():
        _warn("AniGen not cloned. Run: python -m infra.setup.clone")
        return False

    # Step 1: Create venv if needed
    if not venv_py.is_file():
        _log("Creating AniGen venv (Python 3.12, torch cu121)...")
        uv = _uv()
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
        _uv_install(
            venv_py,
            "torch==2.5.0+cu121", "torchvision==0.20.0+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        )

    # Step 2: Basic dependencies from requirements.txt
    _log("Installing AniGen dependencies...")
    _uv_install(venv_py, "wheel", "setuptools>=70.1")
    if (dir_ / "requirements.txt").is_file():
        _uv_install(venv_py, "-r", str(dir_ / "requirements.txt"))

    # Step 3: spconv (pre-built wheel)
    if not _can_import(venv_py, "spconv"):
        _log("Installing spconv...")
        _uv_install(venv_py, "spconv-cu121")

    # Step 4: pytorch3d (compiled)
    if not _can_import(venv_py, "pytorch3d"):
        _log("Building pytorch3d from source...")
        src = _clone_to_tmp("https://github.com/facebookresearch/pytorch3d.git", "pytorch3d", branch="v0.7.8")
        _build_from_source(venv_py, src)

    # Step 5: flash-attn (pre-built)
    if not _can_import(venv_py, "flash_attn"):
        _log("Installing flash-attn...")
        _uv_install(venv_py, "flash-attn", "--no-build-isolation")

    # Step 6: nvdiffrast (compiled)
    if not _can_import(venv_py, "nvdiffrast"):
        _log("Building nvdiffrast for AniGen...")
        src = _clone_to_tmp("https://github.com/NVlabs/nvdiffrast.git", "nvdiffrast_anigen", branch="v0.3.3")
        _build_from_source(venv_py, src)

    # Verify
    if _anigen_ok(venv_py):
        _log(f"AniGen venv ready ({_venv_version(venv_py)})")
        return True
    else:
        _warn("AniGen venv incomplete — some extensions may have failed")
        return False


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


# ─── VibeVoice — multi-speaker TTS ────────────────────────────────────────

def setup_vibevoice() -> bool:
    dir_ = REPOS_DIR / "VibeVoice"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "torch") and _can_import(venv_py, "transformers"):
        _log(f"VibeVoice venv OK ({_venv_version(venv_py)})")
        return True

    if not dir_.is_dir():
        _warn("VibeVoice not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up VibeVoice venv (transformers 4.51.3 pinned)...")
    uv = _uv()
    if not venv_py.is_file():
        _run([uv, "venv", "--python", "3.12", "--quiet"], cwd=str(dir_))
    _uv_install(
        venv_py,
        "torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    )
    _uv_install(venv_py, "transformers==4.51.3", "accelerate", "soundfile")
    # flash-attn for VibeVoice inference
    _uv_install(venv_py, "flash-attn", "--no-build-isolation")

    _log("VibeVoice venv ready")
    return True


# ─── GPT-SoVITS — TTS ─────────────────────────────────────────────────────

def setup_gpt_sovits() -> bool:
    dir_ = REPOS_DIR / "GPT-SoVITS"
    venv_py = dir_ / ".venv" / "bin" / "python"

    if _can_import(venv_py, "torch"):
        _log(f"GPT-SoVITS venv OK ({_venv_version(venv_py)})")
        return True

    if not dir_.is_dir():
        _warn("GPT-SoVITS not cloned. Run: python -m infra.setup.clone")
        return False

    _log("Setting up GPT-SoVITS venv...")
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

    _log("GPT-SoVITS venv ready")
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
    "trellis": setup_trellis,
    "anigen": setup_anigen,
    "ace-step": setup_ace_step,
    "see-through": setup_see_through,
    "vibevoice": setup_vibevoice,
    "gpt-sovits": setup_gpt_sovits,
    "qwen": setup_qwen_img,
    "comfyui": setup_comfyui,
    "llama": setup_llama,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        _log("Setting up all creative tools...")
        for name, fn in TOOLS.items():
            print()
            try:
                fn()
            except Exception as e:
                _warn(f"  {name} setup failed: {e}")
        print()
        _log("All creative tool venvs + llama.cpp build complete.")
        return

    fn = TOOLS.get(target)
    if fn:
        fn()
    else:
        print(f"Usage: python -m infra.setup [{'|'.join(TOOLS)}|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
