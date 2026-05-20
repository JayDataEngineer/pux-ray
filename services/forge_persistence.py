"""Persistence levels for Forge-managed services — controls eviction priority."""
from __future__ import annotations

from enum import IntEnum


class Persistence(IntEnum):
    """Eviction priority — lower values are evicted first.

    TRANSIENT:       Default. Evict freely between calls.
    PERSISTENT:      Prefer to keep loaded. Evict only if no transient services remain.
    PIPELINE_LOCKED: Must stay loaded during pipeline execution. Never evict.
    """
    TRANSIENT = 0
    PERSISTENT = 1
    PIPELINE_LOCKED = 2
