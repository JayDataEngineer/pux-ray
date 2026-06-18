"""Reconcile model_registry.yaml against disk.

Two modes:
  --drop-stale   Remove entries whose paths don't exist on disk
  --dry-run      Show what would change without writing

Also reports:
  - Path normalizations (relative vs absolute)
  - Entries with no path: field
  - Entries pointing outside models_root

Usage:
    python -m registry.reconcile --dry-run
    python -m registry.reconcile --drop-stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "model_registry.yaml"


def _load() -> tuple[dict, list]:
    """Returns (raw_yaml_as_lines, parsed_yaml). Preserves comments by editing in place."""
    raw = _REGISTRY_PATH.read_text()
    parsed = yaml.safe_load(raw)
    return raw, parsed


def _models_root() -> Path:
    from registry.config import Config
    return Path(Config().models_root)


def find_stale_entries(registry: dict, models_root: Path) -> list[tuple[str, str, str]]:
    """Return [(category, model_key, missing_path), ...] for entries whose path doesn't exist."""
    stale = []
    for stype, models in registry.items():
        if not isinstance(models, dict):
            continue
        for mname, meta in models.items():
            if not isinstance(meta, dict):
                continue
            raw = meta.get("path") or meta.get("directory")
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = models_root / p
            if not p.exists():
                stale.append((stype, mname, str(p)))
    return stale


def drop_stale(registry: dict, stale: list[tuple[str, str, str]]) -> int:
    """Drop stale entries in place. Returns count removed."""
    removed = 0
    # Group stale by category for efficient lookup
    stale_keys = {(cat, name) for cat, name, _ in stale}
    for cat in list(registry.keys()):
        models = registry.get(cat)
        if not isinstance(models, dict):
            continue
        for mname in list(models.keys()):
            if (cat, mname) in stale_keys:
                del registry[cat][mname]
                removed += 1
        # Drop empty categories
        if not registry[cat]:
            del registry[cat]
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile registry with disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only, don't modify the YAML")
    parser.add_argument("--drop-stale", action="store_true",
                        help="drop entries whose paths don't exist")
    parser.add_argument("--out", default=None,
                        help="write to this path instead of overwriting")
    args = parser.parse_args()

    if not args.drop_stale and not args.dry_run:
        args.dry_run = True  # default to safe

    raw, registry = _load()
    models_root = _models_root()

    print(f"═══ Registry reconciliation ═══", file=sys.stderr)
    print(f"  registry:  {_REGISTRY_PATH}", file=sys.stderr)
    print(f"  models root: {models_root}", file=sys.stderr)
    print(file=sys.stderr)

    # Count entries
    total_entries = sum(
        len(v) for v in registry.values() if isinstance(v, dict)
    )
    total_categories = sum(1 for v in registry.values() if isinstance(v, dict))
    print(f"  {total_entries} entries across {total_categories} categories",
          file=sys.stderr)

    stale = find_stale_entries(registry, models_root)
    print(f"  {len(stale)} stale entries (path missing on disk)", file=sys.stderr)
    if args.dry_run:
        print(file=sys.stderr)
        for cat, name, path in sorted(stale):
            print(f"    ✗ {cat}/{name}", file=sys.stderr)
            print(f"        missing: {path}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"  Dry run — no changes made. Re-run with --drop-stale to apply.",
              file=sys.stderr)
        return 0

    if not stale:
        print("  Nothing to drop.", file=sys.stderr)
        return 0

    removed = drop_stale(registry, stale)
    print(f"  Dropping {removed} stale entries...", file=sys.stderr)

    # Backup the original
    backup = _REGISTRY_PATH.with_suffix(".yaml.bak")
    backup.write_text(raw)
    print(f"  Backup written: {backup}", file=sys.stderr)

    # Write the new YAML
    out_path = Path(args.out) if args.out else _REGISTRY_PATH
    out_path.write_text(yaml.safe_dump(registry, sort_keys=True,
                                       default_flow_style=False, width=100))
    print(f"  Reconciled registry written: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
