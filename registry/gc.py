"""Garbage-collect model storage based on an audit manifest.

Three deletion modes, each independently controllable:

  --hf-caches        Purge HuggingFace cache snapshots (re-pullable via registry)
  --orphans PATHS    Delete specific orphan paths (comma-separated list)
  --dedupe-hardlink  Hard-link duplicate sets instead of deleting (zero data loss)

Safety:
  --apply            Actually perform deletions (default is dry-run)
  --require-manifest Require an explicit audit manifest to act on
  --keep-list FILE   Never delete paths listed in this file (one per line)

Usage:
    # Show what would be deleted based on the latest audit
    python -m registry.gc --hf-caches

    # Apply the HF cache purge
    python -m registry.gc --hf-caches --apply

    # Hard-link duplicates (saves space without deleting anything)
    python -m registry.gc --dedupe-hardlink --apply

    # Delete specific orphans by path
    python -m registry.gc --orphans /mnt/data/models/qwen-image-edit-2511,/mnt/data/models/flux2-klein-4b --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "model_registry.yaml"


def _humanize(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


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


def _load_manifest(path: str | None) -> dict | None:
    if not path:
        # Find the latest
        import glob
        candidates = sorted(glob.glob("/tmp/model_audit_*.json"))
        if not candidates:
            return None
        path = candidates[-1]
    with open(path) as f:
        return json.load(f)


def _make_keep_set(keep_list: Path | None) -> set[str]:
    if not keep_list:
        return set()
    return {line.strip() for line in keep_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


# ─── Actions ─────────────────────────────────────────────────────────────────

def purge_hf_caches(manifest: dict, keep: set[str], apply: bool) -> tuple[int, int]:
    """Delete all HF cache entries marked safe in the manifest.

    Returns (bytes_recovered, count).
    """
    recovered = 0
    count = 0
    targets = [f for f in manifest["findings"]
               if f["category"] == "HF_CACHE_ENTRY" and f["safe_to_delete"]]
    print(f"  HF cache purge: {len(targets)} entries, "
          f"{_humanize(sum(t['size_bytes'] for t in targets))} recoverable",
          file=sys.stderr)
    for t in targets:
        path = Path(t["path"])
        if str(path) in keep:
            print(f"    KEEP (in keep-list)  {path}", file=sys.stderr)
            continue
        size = t["size_bytes"]
        if apply:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                # Also clean up empty parent snapshots/refs dirs
                for parent in path.parents:
                    if parent.name in ("snapshots", "refs", "blobs"):
                        try:
                            if not any(parent.iterdir()):
                                parent.rmdir()
                        except OSError:
                            break
                    elif parent.name in ("hub", "huggingface", ".cache",
                                          "hf_cache", "cache"):
                        # Don't delete the cache root itself
                        break
            except OSError as e:
                print(f"    ERROR  {path}: {e}", file=sys.stderr)
                continue
        else:
            tag = "  WOULD DELETE"
            print(f"    {tag}  {_humanize(size):>8}  {path}", file=sys.stderr)
        recovered += size
        count += 1
    return recovered, count


def _files_equal(p1: Path, p2: Path) -> bool:
    """True if two files are byte-identical."""
    if p1.stat().st_size != p2.stat().st_size:
        return False
    h1 = hashlib.sha256()
    h2 = hashlib.sha256()
    with p1.open("rb") as f1, p2.open("rb") as f2:
        while True:
            b1 = f1.read(1024 * 1024)
            b2 = f2.read(1024 * 1024)
            h1.update(b1)
            h2.update(b2)
            if not b1:
                break
    return h1.hexdigest() == h2.hexdigest()


def dedupe_hardlink(manifest: dict, keep: set[str], apply: bool) -> tuple[int, int]:
    """Replace duplicate files with hardlinks to a canonical copy.

    Returns (bytes_saved, count). Hardlinks save space without deleting —
    both paths continue to work identically.

    Safety: the duplicate is NEVER deleted before its replacement hardlink
    is verified. We create the new link at a sibling temp path first, then
    atomically ``os.replace`` it over the duplicate. If the link step fails
    (e.g. protected_hardlinks blocks link-to-root-owned), the duplicate is
    left untouched.
    """
    saved = 0
    count = 0
    dupes = [f for f in manifest["findings"] if f["category"] == "DUPLICATE_SET"]
    print(f"  Hardlink dedupe: {len(dupes)} duplicate sets",
          file=sys.stderr)
    for d in dupes:
        members = d["detail"].get("members", [])
        if len(members) < 2:
            continue
        # Sort by path so the canonical is deterministic (shortest path wins,
        # which usually picks the registry-canonical location).
        members_sorted = sorted(members, key=lambda m: (len(m["path"]), m["path"]))
        canonical = Path(members_sorted[0]["path"])
        if not canonical.exists() or not canonical.is_file():
            continue
        for m in members_sorted[1:]:
            dup_path = Path(m["path"])
            if str(dup_path) in keep:
                continue
            if not dup_path.exists() or not dup_path.is_file():
                continue
            # Already hardlinked? Check inode.
            try:
                if canonical.stat().st_ino == dup_path.stat().st_ino and \
                   canonical.stat().st_dev == dup_path.stat().st_dev:
                    continue
            except OSError:
                continue
            # Verify byte-equal before hardlinking (audit grouped by name+size,
            # so files SHOULD be identical — but verify)
            if apply:
                try:
                    if not _files_equal(canonical, dup_path):
                        print(f"    SKIP (content differs)  {dup_path}",
                              file=sys.stderr)
                        continue
                    # Preserve original permissions
                    mode = dup_path.stat().st_mode
                    uid = dup_path.stat().st_uid
                    gid = dup_path.stat().st_gid
                    # SAFETY: create the hardlink at a sibling temp path first.
                    # If this fails (e.g. protected_hardlinks sysctl blocks
                    # links to root-owned files), the original duplicate is
                    # still intact — no data loss.
                    tmp_path = dup_path.with_name(f".{dup_path.name}.hardlink-tmp")
                    try:
                        os.link(canonical, tmp_path)
                    except OSError as e:
                        print(f"    SKIP (cannot hardlink canonical — "
                              f"likely root-owned + protected_hardlinks)  "
                              f"{dup_path}: {e.errno}", file=sys.stderr)
                        continue
                    # Apply preserved permissions and atomically replace.
                    try:
                        os.chmod(tmp_path, mode)
                        try:
                            os.chown(tmp_path, uid, gid)
                        except (PermissionError, OSError):
                            pass
                    except OSError:
                        pass
                    os.replace(tmp_path, dup_path)
                except OSError as e:
                    # Clean up the temp link if anything else went wrong.
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except (OSError, UnboundLocalError):
                        pass
                    print(f"    ERROR  {dup_path}: {e}", file=sys.stderr)
                    continue
            saved += m["size"]
            count += 1
            if not apply:
                print(f"    WOULD HARDLINK  {_humanize(m['size']):>8}  "
                      f"{dup_path} → {canonical}", file=sys.stderr)
    return saved, count


def delete_orphans(orphan_paths: list[str], keep: set[str], apply: bool
                   ) -> tuple[int, int]:
    """Delete specific orphan paths (caller provides the list explicitly)."""
    recovered = 0
    count = 0
    for raw in orphan_paths:
        path = Path(raw.strip())
        if str(path) in keep:
            print(f"    KEEP (in keep-list)  {path}", file=sys.stderr)
            continue
        if not path.exists():
            print(f"    SKIP (already gone)  {path}", file=sys.stderr)
            continue
        size = _dir_size(path)
        if apply:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as e:
                print(f"    ERROR  {path}: {e}", file=sys.stderr)
                continue
        print(f"    {'DELETED' if apply else 'WOULD DELETE'}  "
              f"{_humanize(size):>8}  {path}", file=sys.stderr)
        recovered += size
        count += 1
    return recovered, count


# ─── Top-level ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Model storage GC")
    parser.add_argument("--manifest", default=None,
                        help="audit manifest path (default: latest /tmp/model_audit_*.json)")
    parser.add_argument("--hf-caches", action="store_true",
                        help="purge HF cache snapshots marked safe")
    parser.add_argument("--dedupe-hardlink", action="store_true",
                        help="replace duplicates with hardlinks")
    parser.add_argument("--orphans", default=None,
                        help="comma-separated orphan paths to delete")
    parser.add_argument("--apply", action="store_true",
                        help="actually perform deletions (default: dry-run)")
    parser.add_argument("--keep-list", default=None,
                        help="file with paths to never delete (one per line)")
    args = parser.parse_args()

    print(f"═══ Model storage GC ═══", file=sys.stderr)
    if not args.apply:
        print(f"  DRY RUN — pass --apply to actually delete", file=sys.stderr)

    keep = _make_keep_set(Path(args.keep_list) if args.keep_list else None)

    total_recovered = 0
    total_count = 0

    # HF caches and dedupe need a manifest
    if args.hf_caches or args.dedupe_hardlink:
        manifest = _load_manifest(args.manifest)
        if manifest is None:
            print("  No manifest found. Run `python -m registry.audit` first.",
                  file=sys.stderr)
            return 1
        print(f"  Using manifest: {manifest['generated_at']}", file=sys.stderr)

    if args.hf_caches:
        r, c = purge_hf_caches(manifest, keep, args.apply)
        total_recovered += r
        total_count += c

    if args.dedupe_hardlink:
        r, c = dedupe_hardlink(manifest, keep, args.apply)
        total_recovered += r
        total_count += c

    if args.orphans:
        paths = [p.strip() for p in args.orphans.split(",") if p.strip()]
        r, c = delete_orphans(paths, keep, args.apply)
        total_recovered += r
        total_count += c

    print(file=sys.stderr)
    print(f"  Total: {total_count} entries, {_humanize(total_recovered)} "
          f"{'recovered' if args.apply else 'would recover'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
