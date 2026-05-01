"""Clone or update all creative tool repos and llama.cpp.

Uses git directly (not Docker sidecars) for host-side clones.
Idempotent — skips repos that are already cloned.

Usage:
    python -m infra.setup.clone          # Clone all repos
    python -m infra.setup.clone trellis  # Clone TRELLIS only
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = INFRA_DIR / "repos"

# All repos: (url, directory_name)
REPOS = {
    "trellis": ("https://github.com/microsoft/TRELLIS.2.git", "TRELLIS.2"),
    "anigen": ("https://github.com/VAST-AI-Research/AniGen.git", "AniGen"),
    "ace-step": ("https://github.com/ace-step/ACE-Step-1.5.git", "ACE-Step-1.5"),
    "see-through": ("https://github.com/shitagaki-lab/see-through.git", "see-through"),
    "qwen": ("https://github.com/JayDataEngineer/QwenTTS-Lora-Trainer.git", "qwen_img_expert"),
    "llama": ("https://github.com/ggml-org/llama.cpp.git", "llama.cpp"),
}


def _log(msg: str) -> None:
    print(f"\033[0;32m[clone]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[clone]\033[0m {msg}")


def clone_repo(name: str, url: str, dest: str) -> bool:
    target = REPOS_DIR / dest
    if (target / ".git").is_dir():
        _log(f"{dest} already cloned, pulling latest...")
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(target), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            _warn(f"  git pull failed for {dest}: {result.stderr[:100]}")
        return True

    _log(f"Cloning {dest}...")
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        _warn(f"  git clone failed: {result.stderr[:200]}")
        return False
    return True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        _log("Cloning all tool repos...")
        for name, (url, dest) in REPOS.items():
            try:
                clone_repo(name, url, dest)
            except Exception as e:
                _warn(f"  {name} failed: {e}")
        _log("All repos synced.")
        return

    repo = REPOS.get(target)
    if repo:
        clone_repo(target, repo[0], repo[1])
    else:
        print(f"Usage: python -m infra.setup.clone [{'|'.join(REPOS)}|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
