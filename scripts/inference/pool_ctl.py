#!/usr/bin/env python3
"""pool_ctl — manage the 4-tier inference pool system.

Usage:
  pool_ctl list                  # list all pools (priority order)
  pool_ctl models                # list all routable models
  pool_ctl resolve <model>       # show resolution chain for a model
  pool_ctl status [pool]         # docker + health status (one pool or all)
  pool_ctl start <pool> [model]  # start a pool (optionally via model's launcher)
  pool_ctl stop <pool>           # stop and remove a pool's container
  pool_ctl validate              # check config for warnings
  pool_ctl summary               # one-line per model with optimization
  pool_ctl tier <A|B|C|D>        # list pools in a tier

Examples:
  pool_ctl resolve qwen-image-edit
  pool_ctl start omni-vllm qwen-image-edit
  pool_ctl status omni-vllm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script (no install required).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.inference import PoolManager
from services.inference.launcher import PoolLauncher


def _fmt_status(s: str) -> str:
    if s.startswith("Up"):
        return f"\033[32m{s}\033[0m"   # green
    if s.startswith("Exited"):
        return f"\033[31m{s}\033[0m"   # red
    return f"\033[90m{s}\033[0m"        # grey


def cmd_list(mgr: PoolManager, _args) -> int:
    print(f"{'TIER':<5} {'PRIO':>4}  {'POOL':<14} {'PORT':>5}  {'FRAMEWORK':<12} {'VRAM':>6}  {'MODELS':<6} DESCRIPTION")
    print("-" * 100)
    for p in mgr.pools():
        print(f"{p.tier:<5} {p.priority:>4}  {p.name:<14} {p.port:>5}  {p.framework:<12} "
              f"{p.vram_mb:>5}MB {len(p.models):<6} {p.description}")
    return 0


def cmd_tier(mgr: PoolManager, args) -> int:
    tier = args.tier.upper()
    if tier not in ("A", "B", "C", "D"):
        print(f"Invalid tier: {tier} (use A/B/C/D)", file=sys.stderr)
        return 2
    pools = mgr.system.pools_by_tier(tier)
    if not pools:
        print(f"No pools in tier {tier}")
        return 0
    print(f"═══ Tier {tier} ═══")
    for p in pools:
        print(f"  {p.name:<14} :{p.port}  [{p.framework}]  models={p.models}")
    return 0


def cmd_models(mgr: PoolManager, _args) -> int:
    print(f"{'MODEL':<24} {'PRIMARY':<14} {'FALLBACK':<30} PATH")
    print("-" * 90)
    for m in mgr.models():
        targets = mgr.resolve(m)
        primary = targets[0].pool.name if targets else "-"
        fallback = ",".join(t.pool.name for t in targets[1:]) or "-"
        script = targets[0].launcher.script if (targets and targets[0].launcher) else "-"
        print(f"{m:<24} {primary:<14} {fallback:<30} {script}")
    return 0


def cmd_resolve(mgr: PoolManager, args) -> int:
    targets = mgr.resolve(args.model)
    if not targets:
        print(f"No route for model: {args.model}", file=sys.stderr)
        return 1
    print(f"═══ Resolution chain for {args.model!r} ═══")
    for i, t in enumerate(targets):
        role = "PRIMARY  " if t.is_primary else f"FALLBACK#{t.fallback_index}"
        print(f"  [{t.pool.tier}] {role}  pool={t.pool.name:<14} "
              f":{t.pool.port}  framework={t.pool.framework}")
        if t.launcher:
            if t.launcher.script:
                print(f"           script: {t.launcher.script}")
            if t.launcher.patch:
                print(f"           patch:  {t.launcher.patch}")
            if t.launcher.optimization:
                o = t.launcher.optimization
                opts = [k for k, v in (
                    ("quant", o.quant), ("cache_dit", o.cache_dit),
                    ("taylorseer", o.taylorseer), ("teacache_thresh", o.teacache_thresh),
                    ("vae_tiling", o.vae_tiling), ("cpu_offload_gb", o.cpu_offload_gb),
                ) if v]
                print(f"           opt:    {', '.join(opts)}")
        if t.launcher and t.launcher.benchmark:
            print(f"           bench:")
            for b in t.launcher.benchmark:
                print(f"             - {b.get('steps','?'):>3} steps → {b.get('time_s','?')}s  {b.get('note','')}")
    return 0


def cmd_status(mgr: PoolManager, args) -> int:
    launcher = PoolLauncher(mgr)
    if args.pool:
        print(json.dumps(launcher.status(args.pool), indent=2))
        return 0
    statuses = launcher.status_all()
    print(f"{'POOL':<14} {'TIER':<5} {'PORT':>5}  {'DOCKER':<32} {'HEALTH':<8} FRAMEWORK")
    print("-" * 100)
    for s in statuses:
        ds = _fmt_status(s.get("docker_status", "absent"))
        h = "✓" if s.get("healthy") else "✗"
        print(f"{s['pool']:<14} {s['tier']:<5} {s['port']:>5}  {ds:<60} {h:<8} {s['framework']}")
    return 0


def cmd_start(mgr: PoolManager, args) -> int:
    launcher = PoolLauncher(mgr)
    result = launcher.start(args.pool, model=args.model)
    tag = "✓" if result.healthy else "✗"
    print(f"{tag} {result.pool} (container={result.container}, port={result.port}) "
          f"elapsed={result.elapsed_s:.1f}s healthy={result.healthy}")
    if result.model_loaded:
        print(f"  model loaded: {result.model_loaded}")
    print(f"  {result.message}")
    return 0 if result.healthy else 1


def cmd_stop(mgr: PoolManager, args) -> int:
    launcher = PoolLauncher(mgr)
    ok = launcher.stop(args.pool)
    print(f"{'✓' if ok else '✗'} stopped {args.pool}")
    return 0 if ok else 1


def cmd_validate(mgr: PoolManager, _args) -> int:
    warns = mgr.validate()
    if not warns:
        print("✓ config valid, no warnings")
        return 0
    print(f"⚠ {len(warns)} warning(s):")
    for w in warns:
        print(f"  - {w}")
    return 1


def cmd_summary(mgr: PoolManager, _args) -> int:
    print(f"{'MODEL':<24} {'POOL':<14} {'TIER':<5} {'QUANT':<18} {'CACHE':<6} EXTRA")
    print("-" * 95)
    for m in mgr.models():
        s = mgr.optimization_summary(m)
        if not s.get("served", True):
            continue
        opt_extra = []
        if s.get("teacache_thresh"):
            opt_extra.append(f"teacache={s['teacache_thresh']}")
        if s.get("taylorseer"):
            opt_extra.append("taylorseer")
        print(f"{m:<24} {s['pool']:<14} {s['tier']:<5} "
              f"{(s.get('quant') or '-'):<18} "
              f"{'✓' if s.get('cache_dit') else '-':<6} "
              f"{', '.join(opt_extra)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tech Noir inference pool control")
    parser.add_argument("--config", default=None, help="path to inference_pools.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all pools").set_defaults(func=cmd_list)
    sub.add_parser("models", help="list all routable models").set_defaults(func=cmd_models)
    sub.add_parser("validate", help="check config").set_defaults(func=cmd_validate)
    sub.add_parser("summary", help="model optimization summary").set_defaults(func=cmd_summary)

    p_tier = sub.add_parser("tier", help="list pools in a tier (A/B/C/D)")
    p_tier.add_argument("tier")
    p_tier.set_defaults(func=cmd_tier)

    p_resolve = sub.add_parser("resolve", help="show resolution chain for a model")
    p_resolve.add_argument("model")
    p_resolve.set_defaults(func=cmd_resolve)

    p_start = sub.add_parser("start", help="start a pool")
    p_start.add_argument("pool")
    p_start.add_argument("model", nargs="?", help="use this model's launch script")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="stop a pool's container")
    p_stop.add_argument("pool")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="docker + health status")
    p_status.add_argument("pool", nargs="?")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        mgr = PoolManager.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return args.func(mgr, args)


if __name__ == "__main__":
    sys.exit(main())
