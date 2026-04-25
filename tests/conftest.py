"""Test configuration and fixtures for Ray integration tests."""

import subprocess
import pytest


def ray_is_running() -> bool:
    try:
        result = subprocess.run(
            ["ray", "status"], capture_output=True, text=True, timeout=5,
        )
        return "node" in result.stdout
    except Exception:
        return False


@pytest.fixture(scope="session")
def ray_cluster():
    """Ensure Ray cluster is running for the test session."""
    if not ray_is_running():
        subprocess.run(
            ["bash", "scripts/start_cluster.sh"],
            check=True, timeout=30,
        )
    yield
    # Don't stop the cluster after tests - leave it for manual inspection


@pytest.fixture
def free_vram_mb() -> int:
    """Get current free VRAM in MB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    return int(result.stdout.strip().split("\n")[0].strip())
