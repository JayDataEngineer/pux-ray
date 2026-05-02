"""Tech Noir Ray — System provisioning.

Configures the host OS for AI workload requirements:
- apt packages (gcc-14 for CUDA 12.x builds, build tools)
- sysctl tuning (vm.overcommit_memory for Ray/PyTorch)
- Swap file on /mnt/data (if LUKS data mount exists)

Usage:
    sudo python -m infra.setup system          # Run all checks
    sudo python -m infra.setup system --fix     # Apply fixes

Must run as root (uses sudo internally for individual commands).
Idempotent — safe to re-run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(f"\033[0;32m[system]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[system]\033[0m {msg}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# ---------------------------------------------------------------------------
# apt packages
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "gcc-14", "g++-14",          # CUDA 12.x requires GCC <= 14 (Ubuntu 26.04 has 15)
    "build-essential", "cmake", "ninja-build", "patch",
    "python3-dev", "python3-venv",
]


def check_apt_packages(fix: bool = False) -> list[str]:
    """Check/install required apt packages."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        result = _run(["dpkg", "-s", pkg], timeout=10)
        if result.returncode != 0:
            missing.append(pkg)

    if not missing:
        _log("All required apt packages installed")
        return missing

    _warn(f"Missing apt packages: {', '.join(missing)}")
    if fix:
        _log(f"Installing {len(missing)} packages...")
        result = _run(
            ["sudo", "apt", "install", "-y"] + missing,
            timeout=300,
        )
        if result.returncode != 0:
            _warn(f"apt install failed: {result.stderr[-300:]}")
        else:
            _log("Packages installed")
    return missing


# ---------------------------------------------------------------------------
# sysctl: vm.overcommit_memory
# ---------------------------------------------------------------------------

SYSCTL_FILE = Path("/etc/sysctl.d/99-tech-noir-overcommit.conf")


def check_sysctl(fix: bool = False) -> bool:
    """Ensure vm.overcommit_memory=1 for Ray/PyTorch memory allocation."""
    result = _run(["sysctl", "-n", "vm.overcommit_memory"], timeout=5)
    current = result.stdout.strip() if result.returncode == 0 else "unknown"

    if current == "1":
        _log("vm.overcommit_memory = 1 (OK)")
        return True

    _warn(f"vm.overcommit_memory = {current} (expected 1)")
    if fix:
        _log("Setting vm.overcommit_memory=1...")
        SYSCTL_FILE.write_text("# Tech Noir Ray — allow memory overcommit for PyTorch/Ray\n"
                               "vm.overcommit_memory=1\n")
        _run(["sudo", "sysctl", "--system"], timeout=10)
        _log("sysctl configured")
    return False


# ---------------------------------------------------------------------------
# Swap file on /mnt/data
# ---------------------------------------------------------------------------

DATA_MOUNT = Path("/mnt/data")
SWAP_FILE = DATA_MOUNT / "swapfile"
SWAP_SIZE_GB = 64


def check_swap(fix: bool = False) -> bool:
    """Check/create swap file on /mnt/data."""
    if not DATA_MOUNT.is_mount():
        _warn("/mnt/data not mounted — skipping swap setup")
        return False

    result = _run(["swapon", "--show=NAME", "--noheadings"], timeout=5)
    active_swaps = result.stdout.strip().split("\n") if result.stdout.strip() else []

    if str(SWAP_FILE) in active_swaps:
        _log(f"Swap active: {SWAP_FILE}")
        return True

    if SWAP_FILE.exists():
        _warn(f"Swap file exists but not active: {SWAP_FILE}")
        if fix:
            _run(["sudo", "swapon", str(SWAP_FILE)], timeout=10)
            _log("Swap activated")
            return True
        return False

    _warn(f"No swap file at {SWAP_FILE}")
    if fix:
        _log(f"Creating {SWAP_SIZE_GB}GB swap file...")
        cmds = [
            ["sudo", "fallocate", "-l", f"{SWAP_SIZE_GB}G", str(SWAP_FILE)],
            ["sudo", "chmod", "600", str(SWAP_FILE)],
            ["sudo", "mkswap", str(SWAP_FILE)],
            ["sudo", "swapon", str(SWAP_FILE)],
        ]
        for cmd in cmds:
            result = _run(cmd, timeout=60)
            if result.returncode != 0:
                _warn(f"Failed: {' '.join(cmd)}: {result.stderr[:200]}")
                return False

        fstab = Path("/etc/fstab").read_text()
        if str(SWAP_FILE) not in fstab:
            fstab_entry = f"{SWAP_FILE} none swap sw 0 0\n"
            _run(["sudo", "sh", "-c", f"echo '{fstab_entry.strip()}' >> /etc/fstab"], timeout=10)
            _log("Added swap to /etc/fstab")

        _log(f"Swap created and activated: {SWAP_FILE} ({SWAP_SIZE_GB}GB)")
        return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fix = "--fix" in sys.argv

    if os.geteuid() != 0 and not fix:
        _log("Running in check mode. Use --fix to apply changes.")

    _log("=" * 60)
    _log("Tech Noir System Provisioning")
    _log("=" * 60)

    issues = 0

    missing = check_apt_packages(fix=fix)
    if missing:
        issues += len(missing)

    if not check_sysctl(fix=fix):
        issues += 1

    if not check_swap(fix=fix):
        issues += 1

    _log("=" * 60)
    if issues == 0:
        _log("All system checks passed!")
    elif fix:
        _log(f"Applied fixes for {issues} issues. Re-run to verify.")
    else:
        _log(f"{issues} issues found. Run with --fix to apply fixes.")

    return 0 if issues == 0 or fix else 1


if __name__ == "__main__":
    sys.exit(main())
