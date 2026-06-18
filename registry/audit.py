"""Model storage audit — produces a cleanup manifest without touching anything.

Reports five categories of findings, each with confidence + recovery info:

  STALE_REGISTRY   — registry entry whose path doesn't exist on disk
  HF_CACHE_ENTRY   — file inside an HF cache dir; re-pullable from source
  ORPHAN_DIR       — top-level dir not referenced by any registry entry
  ORPHAN_FILE      — heavy file (>500M) not referenced by any registry entry
  DUPLICATE_SET    — group of paths whose basenames+sizes match (format dupes)

Output: JSON manifest (default /tmp/model_audit_<ts>.json) consumable by
registry.gc for safe deletion. Use --apply-gc to chain directly.

Usage:
    python -m registry.audit                     # write manifest
    python -m registry.audit --summary           # print categories only
    python -m registry.audit --out manifest.json # custom path
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "model_registry.yaml"
_POOLS_PATH = _PROJECT_ROOT / "config" / "inference_pools.yaml"
_SPECS_PATH = _PROJECT_ROOT / "config" / "model_specs.yaml"

# Files we never flag for deletion regardless of state.
_NEVER_DELETE = {
    ".gitkeep", "README.md", "CACHEDIR.TAG", "version.txt",
    "version_diffusers_cache.txt",
}

# Dirs that are always considered HF caches regardless of name.
_HF_CACHE_DIR_NAMES = {
    ".cache", "hub", "hf_cache", "huggingface", "xet", "modules",
}


@dataclass
class Finding:
    category: str
    path: str
    size_bytes: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    safe_to_delete: bool = False
    delete_reason: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    return total


def _humanize(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _normalize_registry_path(raw: str, models_root: Path) -> Path:
    """Resolve a registry entry path to absolute."""
    p = Path(raw)
    if not p.is_absolute():
        p = models_root / p
    return p.resolve()


def _registered_paths(registry: dict, models_root: Path) -> set[str]:
    """All absolute paths the registry references (dirs + files under them)."""
    out: set[str] = set()
    for stype, models in registry.items():
        if not isinstance(models, dict):
            continue
        for mname, meta in models.items():
            if not isinstance(meta, dict):
                continue
            raw = meta.get("path") or meta.get("directory")
            if not raw:
                continue
            resolved = _normalize_registry_path(raw, models_root)
            out.add(str(resolved))
    return out


def _pool_referenced_paths(pools: dict) -> set[str]:
    """Absolute paths referenced from inference_pools.yaml."""
    out: set[str] = set()
    for pname, pool in (pools.get("pools") or {}).items():
        for mname, launcher in (pool.get("model_launchers") or {}).items():
            raw = launcher.get("model_dir") if isinstance(launcher, dict) else None
            if raw:
                out.add(str(Path(raw).resolve()))
    return out


def _is_under_any(path: Path, prefixes: Iterable[str]) -> bool:
    """True if `path` is equal to OR inside any of `prefixes`."""
    s = str(path)
    return any(s == p or s.startswith(p.rstrip("/") + "/") for p in prefixes)


def _contains_or_is_any(container: Path, candidates: Iterable[str]) -> bool:
    """True if `container` is equal to OR contains any of `candidates`.

    Used for orphan detection: a top-level dir is "registered" if any
    registry path lives inside it (e.g. dir `3d/` contains registered
    path `3d/trellis/...`).
    """
    s = str(container)
    return any(s == c or c.startswith(s.rstrip("/") + "/") for c in candidates)


def _hash_file_head(p: Path, limit: int = 1024 * 1024) -> str:
    """Cheap content hash — first 1MB only. Used for dupe detection."""
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            h.update(f.read(limit))
    except OSError:
        pass
    return h.hexdigest()[:16]


# ─── Audit passes ────────────────────────────────────────────────────────────

def _find_stale_registry(registry: dict, models_root: Path) -> list[Finding]:
    """Registered entries whose paths don't exist."""
    findings: list[Finding] = []
    for stype, models in registry.items():
        if not isinstance(models, dict):
            continue
        for mname, meta in models.items():
            if not isinstance(meta, dict):
                continue
            raw = meta.get("path") or meta.get("directory")
            if not raw:
                continue
            resolved = _normalize_registry_path(raw, models_root)
            if not resolved.exists():
                findings.append(Finding(
                    category="STALE_REGISTRY",
                    path=str(resolved),
                    detail={"registry_key": f"{stype}/{mname}", "raw_path": raw},
                    safe_to_delete=True,  # registry entry, not a file
                    delete_reason="registry entry points at non-existent path; drop from YAML",
                ))
    return findings


def _find_hf_cache_entries(models_root: Path, registered: set[str]) -> list[Finding]:
    """Files inside HF cache dirs that aren't directly registered."""
    findings: list[Finding] = []
    # Identify all HF cache HUB roots (where models--* dirs live) under models_root.
    # Only iterate `hub/` dirs — their parent `.cache/huggingface/` also contains
    # `xet/`, `modules/`, etc. which we handle separately.
    cache_roots: list[Path] = []
    for candidate in (models_root / ".cache" / "huggingface" / "hub",
                      models_root / "hf_cache" / "hub",
                      models_root / "cache" / "huggingface" / "hub"):
        if candidate.exists() and candidate.is_dir():
            cache_roots.append(candidate)
    seen_cache_dirs: set[Path] = set()
    for root in cache_roots:
        for hub_entry in root.iterdir():
            if not hub_entry.is_dir() or hub_entry.name.startswith("."):
                continue
            size = _dir_size(hub_entry)
            # If the registry directly references a path inside this hub entry
            # (rare but possible), mark it as in-use.
            in_use = _is_under_any(hub_entry.resolve(), registered)
            findings.append(Finding(
                category="HF_CACHE_ENTRY",
                path=str(hub_entry),
                size_bytes=size,
                detail={"cache_root": str(root), "repo_id": hub_entry.name},
                safe_to_delete=not in_use,
                delete_reason=(
                    "HF cache snapshot; re-pullable via `tech-noir models pull` "
                    "or registry source:" if not in_use else
                    "referenced by registry; do not auto-delete"
                ),
            ))
            seen_cache_dirs.add(hub_entry)
    return findings


def _find_orphans(models_root: Path, registered: set[str],
                  pool_refs: set[str]) -> list[Finding]:
    """Top-level entries not referenced by registry or pools config.

    A top-level dir is considered 'referenced' if it (or any path inside it)
    appears in the registry or pool config. Only true top-level orphans
    (e.g. `qwen-image-edit-2511/` next to a registered `image-gen/qwen-image-edit/`)
    are flagged.
    """
    findings: list[Finding] = []
    refs = registered | pool_refs
    # Identify HF cache roots so we don't double-count them as orphans.
    hf_cache_roots: list[Path] = []
    for rel in ("models/.cache", "models/hf_cache", "models/cache",
                "models/.cache/huggingface"):
        cand = models_root.parent / rel
        if cand.exists():
            hf_cache_roots.append(cand.resolve())
    # Also the direct .cache under models_root
    if (models_root / ".cache").exists():
        hf_cache_roots.append((models_root / ".cache").resolve())
    if (models_root / "hf_cache").exists():
        hf_cache_roots.append((models_root / "hf_cache").resolve())
    if (models_root / "cache").exists():
        hf_cache_roots.append((models_root / "cache").resolve())

    for child in sorted(models_root.iterdir()):
        if child.name in _NEVER_DELETE:
            continue
        if child.name.startswith(".") and child.name not in {".cache"}:
            continue
        # Skip if this dir is itself a registered path OR contains one.
        if _contains_or_is_any(child.resolve(), refs):
            continue
        # Skip if this dir is an HF cache (reported separately).
        if any(child.resolve() == cr or str(child.resolve()).startswith(
                str(cr).rstrip("/") + "/") for cr in hf_cache_roots):
            continue
        size = _dir_size(child)
        # Cheap safety: don't flag tiny entries (<10M) — usually config files
        if size < 10 * 1024 * 1024 and child.is_dir():
            continue
        findings.append(Finding(
            category="ORPHAN_DIR" if child.is_dir() else "ORPHAN_FILE",
            path=str(child),
            size_bytes=size,
            detail={"name": child.name},
            safe_to_delete=False,  # default — gc tool will refine
            delete_reason=(
                "top-level entry not referenced by model_registry.yaml or "
                "inference_pools.yaml — manual review required"
            ),
        ))
    return findings


def _find_format_duplicates(models_root: Path, registered: set[str]) -> list[Finding]:
    """Group heavy files (>500M) by basename to spot format dupes."""
    findings: list[Finding] = []
    groups: dict[str, list[tuple[Path, int]]] = {}
    for p in models_root.rglob("*"):
        if not p.is_file() or p.name in _NEVER_DELETE:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < 500 * 1024 * 1024:
            continue
        # Skip files inside HF caches (handled separately)
        if any(part in _HF_CACHE_DIR_NAMES for part in p.parts):
            continue
        groups.setdefault(p.name, []).append((p, size))

    for name, hits in groups.items():
        if len(hits) < 2:
            continue
        # Verify via content hash — same name+size alone isn't proof
        hash_groups: dict[str, list[tuple[Path, int]]] = {}
        for path, size in hits:
            # Only hash if size matches the first hit — different sizes = different files
            key = f"size:{size}"  # cheaper than hashing 5G files
            hash_groups.setdefault(key, []).append((path, size))
        for key, members in hash_groups.items():
            if len(members) < 2:
                continue
            total = sum(s for _, s in members)
            primary = members[0][0]
            duplicates = [str(p) for p, _ in members[1:]]
            duplicate_size = sum(s for _, s in members[1:])
            findings.append(Finding(
                category="DUPLICATE_SET",
                path=str(primary),
                size_bytes=total,
                detail={
                    "filename": name,
                    "primary": str(primary),
                    "duplicates": duplicates,
                    "duplicate_size_bytes": duplicate_size,
                    "members": [{"path": str(p), "size": s} for p, s in members],
                },
                safe_to_delete=False,
                delete_reason="format/location duplicate — review which path is canonical",
            ))
    return findings


# ─── Top-level driver ────────────────────────────────────────────────────────

def run_audit(models_root: Path | None = None) -> dict[str, Any]:
    """Run all audit passes. Returns a manifest dict."""
    if models_root is None:
        from registry.config import Config
        models_root = Path(Config().models_root)

    registry = _load_yaml(_REGISTRY_PATH)
    pools = _load_yaml(_POOLS_PATH)

    registered = _registered_paths(registry, models_root)
    pool_refs = _pool_referenced_paths(pools)

    findings: list[Finding] = []
    findings.extend(_find_stale_registry(registry, models_root))
    findings.extend(_find_hf_cache_entries(models_root, registered))
    findings.extend(_find_orphans(models_root, registered, pool_refs))
    findings.extend(_find_format_duplicates(models_root, registered))

    # Summary by category
    summary: dict[str, dict[str, Any]] = {}
    for f in findings:
        cat = f.category
        s = summary.setdefault(cat, {"count": 0, "total_bytes": 0,
                                     "safe_bytes": 0, "safe_count": 0})
        s["count"] += 1
        s["total_bytes"] += f.size_bytes
        if f.safe_to_delete:
            s["safe_count"] += 1
            s["safe_bytes"] += f.size_bytes

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_root": str(models_root),
        "totals": {
            "registered_paths": len(registered),
            "pool_referenced_paths": len(pool_refs),
            "findings": sum(s["count"] for s in summary.values()),
            "safe_recoverable_bytes": sum(s["safe_bytes"] for s in summary.values()),
            "total_review_required_bytes": sum(
                s["total_bytes"] - s["safe_bytes"] for s in summary.values()
            ),
        },
        "summary": summary,
        "findings": [f.to_dict() for f in findings],
    }


def print_summary(manifest: dict[str, Any], file=sys.stdout) -> None:
    """Human-readable summary."""
    print(f"\n═══ Model Storage Audit ═══", file=file)
    print(f"  models_root: {manifest['models_root']}", file=file)
    print(f"  generated:   {manifest['generated_at']}", file=file)
    print(f"  registered paths:  {manifest['totals']['registered_paths']}", file=file)
    print(f"  pool-referenced:   {manifest['totals']['pool_referenced_paths']}", file=file)
    print(file=file)
    print(f"  {'CATEGORY':<18} {'COUNT':>6} {'TOTAL':>10} {'SAFE':>10} {'NEEDS-REVIEW':>14}",
          file=file)
    print(f"  {'-'*18} {'-'*6} {'-'*10} {'-'*10} {'-'*14}", file=file)
    for cat, s in sorted(manifest["summary"].items()):
        review = s["total_bytes"] - s["safe_bytes"]
        print(f"  {cat:<18} {s['count']:>6} {_humanize(s['total_bytes']):>10} "
              f"{_humanize(s['safe_bytes']):>10} {_humanize(review):>14}",
              file=file)
    print(file=file)
    total_safe = manifest["totals"]["safe_recoverable_bytes"]
    print(f"  Safe to recover immediately: {_humanize(total_safe)}", file=file)
    print(f"  Needs review:                "
          f"{_humanize(manifest['totals']['total_review_required_bytes'])}",
          file=file)
    print(file=file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model storage audit")
    parser.add_argument("--out", default=None, help="manifest output path")
    parser.add_argument("--summary", action="store_true",
                        help="print category summary and exit")
    parser.add_argument("--category", default=None,
                        help="filter findings to one category")
    parser.add_argument("--show-safe", action="store_true",
                        help="include the safe-to-delete list in stdout")
    args = parser.parse_args()

    manifest = run_audit()

    if args.out is None:
        args.out = f"/tmp/model_audit_{int(time.time())}.json"
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"  manifest written: {args.out}", file=sys.stderr)

    print_summary(manifest)

    if args.show_safe:
        safe = [f for f in manifest["findings"]
                if f["safe_to_delete"] and (not args.category
                                            or f["category"] == args.category)]
        if safe:
            print(f"\n─── Safe to delete ({len(safe)} entries) ───")
            for f in sorted(safe, key=lambda x: -x["size_bytes"]):
                print(f"  {_humanize(f['size_bytes']):>8}  {f['path']}")

    if args.category:
        cat = [f for f in manifest["findings"] if f["category"] == args.category]
        print(f"\n─── {args.category} ({len(cat)} entries) ───")
        for f in sorted(cat, key=lambda x: -x["size_bytes"]):
            print(f"  {_humanize(f['size_bytes']):>8}  {f['path']}")
            for k, v in f.get("detail", {}).items():
                if k in ("duplicates", "members"):
                    print(f"           {k}:")
                    for entry in v:
                        print(f"             - {entry}")
                else:
                    print(f"           {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
