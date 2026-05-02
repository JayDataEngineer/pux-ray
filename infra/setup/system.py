"""Tech Noir Ray — System provisioning.

Configures the host OS for AI workload requirements:
- apt packages (gcc-14 for CUDA 12.x builds, build tools)
- CUDA header patches (glibc 2.41+ noexcept fix)
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

INFRA_DIR = Path(__file__).resolve().parent
RAY_ROOT = INFRA_DIR.parent.parent


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
# CUDA header patch (glibc 2.41+ noexcept fix)
# ---------------------------------------------------------------------------

def check_cuda_header_patch(fix: bool = False) -> bool:
    """Patch CUDA math_functions.h to add noexcept to rsqrt/rsqrtf.

    Ubuntu 26.04 (glibc 2.42) declares rsqrt/rsqrtf with noexcept in
    <bits/mathcalls.h>, but CUDA's math_functions.h doesn't match.
    This causes compile errors when building CUDA extensions.

    Affects: CUDA 12.x and 13.x on Ubuntu 26.04+.
    """
    patched = False
    for cuda_home in ["/usr/local/cuda-13.1", "/usr/local/cuda-12.8"]:
        header = Path(cuda_home) / "include" / "crt" / "math_functions.h"
        if not header.exists():
            continue

        content = header.read_text()
        if "rsqrt(double x) noexcept" in content:
            _log(f"CUDA header already patched: {header}")
            patched = True
            continue

        _warn(f"CUDA header needs noexcept patch: {header}")
        if fix:
            # Add noexcept to rsqrt and rsqrtf declarations
            new_content = content.replace(
                'rsqrt(double x);\n',
                'rsqrt(double x) noexcept;\n',
            ).replace(
                'rsqrtf(float x);\n',
                'rsqrtf(float x) noexcept;\n',
            )
            if new_content != content:
                # Backup and write
                backup = header.with_suffix(header.suffix + ".orig")
                if not backup.exists():
                    import shutil
                    shutil.copy2(header, backup)
                _run(["sudo", "tee", str(header)], input=new_content, timeout=10)
                _log(f"Patched {header}")
                patched = True
            else:
                _warn(f"Could not find rsqrt declarations to patch in {header}")
    return patched


# ---------------------------------------------------------------------------
# sysctl: vm.overcommit_memory
# ---------------------------------------------------------------------------

SYSCTL_FILE = Path("/etc/sysctl.d/99-tech-noir-overcommit.conf")


def check_sysctl(fix: bool = False) -> bool:
    """Ensure vm.overcommit_memory=1 for Ray/PyTorch memory allocation."""
    # Check current value
    result = _run(["sysctl", "-n", "vm.overcommit_memory"], timeout=5)
    current = result.stdout.strip() if result.returncode == 0 else "unknown"

    if current == "1":
        _log(f"vm.overcommit_memory = 1 (OK)")
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
    # Check if /mnt/data is mounted
    if not DATA_MOUNT.is_mount():
        _warn("/mnt/data not mounted — skipping swap setup")
        return False

    # Check if swap file is active
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

        # Add to fstab if not already there
        fstab = Path("/etc/fstab").read_text()
        if str(SWAP_FILE) not in fstab:
            fstab_entry = f"{SWAP_FILE} none swap sw 0 0\n"
            Path("/tmp/fstab_swap").write_text(fstab_entry)
            _run(["sudo", "sh", "-c", f"echo '{fstab_entry.strip()}' >> /etc/fstab"], timeout=10)
            _log("Added swap to /etc/fstab")

        _log(f"Swap created and activated: {SWAP_FILE} ({SWAP_SIZE_GB}GB)")
        return True

    return False


# ---------------------------------------------------------------------------
# torch CUDA version check bypass
# ---------------------------------------------------------------------------

def check_torch_cuda_patch(venv_dir: Path, fix: bool = False) -> bool:
    """Patch torch _check_cuda_version to skip CUDA version mismatch.

    CUDA 13.1 on the system vs torch cu124/cu128 triggers a version
    check error during extension builds. We bypass it by returning early.
    """
    cpp_ext = venv_dir / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "utils" / "cpp_extension.py"
    if not cpp_ext.exists():
        _warn(f"torch cpp_extension.py not found: {cpp_ext}")
        return False

    content = cpp_ext.read_text()
    marker = "return  # Patched: skip CUDA version check"

    if marker in content:
        _log("torch _check_cuda_version already patched")
        return True

    _warn(f"torch _check_cuda_version needs patching: {cpp_ext}")
    if fix:
        # Insert 'return' as first line of _check_cuda_version function
        old = "def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:\n"
        new = old + f"    {marker}\n"
        if old in content:
            cpp_ext.write_text(content.replace(old, new))
            _log("Patched torch _check_cuda_version")
            return True
        else:
            _warn("Could not find _check_cuda_version function signature")
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

    # 1. apt packages
    missing = check_apt_packages(fix=fix)
    if missing:
        issues += len(missing)

    # 2. CUDA header patch
    if not check_cuda_header_patch(fix=fix):
        issues += 1

    # 3. sysctl
    if not check_sysctl(fix=fix):
        issues += 1

    # 4. swap
    if not check_swap(fix=fix):
        issues += 1

    # 5. torch CUDA patch (for TRELLIS venv)
    trellis_dir = RAY_ROOT / "infra" / "repos" / "TRELLIS.2"
    if trellis_dir.exists():
        if not check_torch_cuda_patch(trellis_dir, fix=fix):
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
