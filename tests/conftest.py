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
    """Get current free VRAM in MB via torch.cuda (Ray-native, no nvidia-smi)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            return int(total - reserved)
    except Exception:
        pass
    return 0
