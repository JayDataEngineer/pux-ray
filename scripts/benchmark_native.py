#!/usr/bin/env python3
"""
Phase 1 Benchmark: Native diffusers optimization paths on RTX 4090.

Tests different optimization stacks against each other to get REAL numbers
instead of theoretical estimates. Designed to run on the worker pod.

Usage:
  python benchmark_native.py --model flux-schnell --path baseline
  python benchmark_native.py --model flux-schnell --path compile-cache
  python benchmark_native.py --model flux-schnell --path group-offload
  python benchmark_native.py --model flux-schnell --path all  (runs every path)
  python benchmark_native.py --model flux-schnell --path all --save-output

Models supported:
  flux-schnell   — Standard diffusers pipeline (simplest, proves the concept)
  flux-dev       — Same pipeline, 20 steps (needs HF auth for dev)

Outputs:
  - Per-phase timing (text encoding / denoising / VAE decode)
  - Peak VRAM (allocated, reserved, system)
  - Steps/second
  - Optional: saved PNG for visual quality comparison
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODELS = {
    "flux-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "pipeline_cls": "FluxPipeline",
        "default_steps": 4,
        "default_guidance": 0.0,
        "default_size": (1024, 1024),
        "prompt": "a cinematic photo of a golden retriever puppy in a field of wildflowers, golden hour lighting, shallow depth of field",
        "license": "Apache-2.0",
    },
    "flux-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline_cls": "FluxPipeline",
        "default_steps": 20,
        "default_guidance": 3.5,
        "default_size": (1024, 1024),
        "prompt": "a cinematic photo of a golden retriever puppy in a field of wildflowers, golden hour lighting, shallow depth of field",
        "license": "Non-commercial (needs HF token)",
    },
}


# ---------------------------------------------------------------------------
# VRAM utilities
# ---------------------------------------------------------------------------
def reset_vram():
    """Clear caches and reset peak counters."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb():
    """Get current VRAM usage in MB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def get_system_vram_mb():
    """Get system-level VRAM from nvidia-smi (includes CUDA context overhead)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def get_gpu_temp():
    """Get GPU temperature for throttling detection."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(", ")
        return {
            "temp_c": int(parts[0]),
            "sm_clock_mhz": int(parts[1]),
            "max_sm_clock_mhz": int(parts[2]),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model loading for each optimization path
# ---------------------------------------------------------------------------
def load_baseline(model_cfg):
    """Baseline: model_cpu_offload (component-level swap), no other optimizations."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe, "baseline (model_cpu_offload, no compile/cache/offload)"


def load_compile_only(model_cfg):
    """Path A (compile only): model_cpu_offload + compile_repeated_blocks. NO cache."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    compile_status = "no compile"
    try:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        compile_status = "compile_repeated_blocks: ON"
    except Exception as e:
        compile_status = f"compile_repeated_blocks: FAILED ({e})"

    return pipe, f"compile_only: model_cpu_offload + {compile_status}"


def load_cache_only(model_cfg):
    """Path A (cache only): model_cpu_offload + first_block_cache. NO compile."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    try:
        from diffusers import apply_first_block_cache, FirstBlockCacheConfig
        apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.05))
        cache_status = "first_block_cache(threshold=0.05): ON"
    except Exception as e:
        cache_status = f"first_block_cache: FAILED ({e})"

    return pipe, f"cache_only: model_cpu_offload + {cache_status}"


def load_group_offload(model_cfg):
    """Path B: layerwise_casting + group_offload (no compile, no cache)."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )

    # Move text encoders + VAE to GPU (they need to be resident)
    pipe.text_encoder.to("cuda")
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()

    # Layerwise casting FIRST (FP8 storage, bf16 compute)
    try:
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        cast_status = "layerwise_casting(fp8): ON"
    except Exception as e:
        cast_status = f"layerwise_casting: FAILED ({e})"

    # Group offload SECOND
    # NOTE: use_stream=True forces num_blocks_per_group=1 in diffusers 0.37.0
    try:
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,   # use_stream requires this in 0.37.0
            use_stream=True,
            record_stream=True,
        )
        offload_status = f"group_offload(stream, blocks=1): ON"
    except Exception as e:
        offload_status = f"group_offload: FAILED ({e})"

    return pipe, f"Path B: {cast_status} + {offload_status} + vae_tiling"


LOADERS = {
    "baseline": load_baseline,
    "compile-only": load_compile_only,
    "cache-only": load_cache_only,
    "group-offload": load_group_offload,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(pipe, model_cfg, num_warmup=3, num_timed=5, save_output=False, output_dir=None):
    """Run benchmark with warmup + timed runs. Returns metrics dict."""

    prompt = model_cfg["prompt"]
    steps = model_cfg["default_steps"]
    guidance = model_cfg["default_guidance"]
    width, height = model_cfg["default_size"]

    results = {
        "warmup": [],
        "timed": [],
    }

    # === WARMUP (results discarded) ===
    for i in range(num_warmup):
        print(f"  [warmup {i+1}/{num_warmup}]", end="", flush=True)
        t0 = time.perf_counter()
        _ = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.2f}s")
        results["warmup"].append(elapsed)
        reset_vram()

    # === TIMED RUNS ===
    for i in range(num_timed):
        reset_vram()
        vram_before = get_system_vram_mb()

        print(f"  [timed {i+1}/{num_timed}]", end="", flush=True)

        t_total_start = time.perf_counter()

        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            return_dict=True,
        )

        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total_start

        vram_metrics = get_vram_mb()
        vram_after = get_system_vram_mb()
        temp = get_gpu_temp()

        print(f"  {t_total:.2f}s  peak_vram={vram_metrics['peak_allocated_mb']:.0f}MB")

        results["timed"].append({
            "total_s": t_total,
            "steps": steps,
            "steps_per_second": steps / t_total,
            "vram_peak_allocated_mb": vram_metrics["peak_allocated_mb"],
            "vram_peak_reserved_mb": vram_metrics["peak_reserved_mb"],
            "vram_system_before_mb": vram_before,
            "vram_system_after_mb": vram_after,
            "gpu_temp_c": temp["temp_c"] if temp else None,
            "gpu_sm_clock_mhz": temp["sm_clock_mhz"] if temp else None,
            "gpu_max_sm_clock_mhz": temp["max_sm_clock_mhz"] if temp else None,
        })

        # Save first timed output for visual comparison
        if save_output and i == 0 and output_dir:
            img = output.images[0]
            outpath = output_dir / f"benchmark_output.png"
            img.save(outpath)
            print(f"  Saved: {outpath}")

        # Cooldown between runs (prevent thermal throttling)
        if i < num_timed - 1:
            time.sleep(3)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def compute_stats(runs):
    """Compute mean/std from timed runs."""
    totals = [r["total_s"] for r in runs]
    vrams = [r["vram_peak_allocated_mb"] for r in runs]
    sps = [r["steps_per_second"] for r in runs]

    n = len(totals)
    mean_total = sum(totals) / n
    mean_vram = sum(vrams) / n
    mean_sps = sum(sps) / n

    if n > 1:
        var_total = sum((t - mean_total) ** 2 for t in totals) / (n - 1)
        std_total = var_total ** 0.5
    else:
        std_total = 0.0

    return {
        "mean_total_s": mean_total,
        "std_total_s": std_total,
        "min_total_s": min(totals),
        "max_total_s": max(totals),
        "mean_peak_vram_mb": mean_vram,
        "mean_steps_per_second": mean_sps,
        "n_runs": n,
    }


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 90)
    print(f"{'Path':<25} {'Mean (s)':<12} {'Std':<10} {'Min (s)':<12} {'Peak VRAM':<15} {'Steps/s':<10}")
    print("-" * 90)

    for path_name, data in all_results.items():
        if data is None:
            print(f"{path_name:<25} {'FAILED':<12}")
            continue
        s = data["stats"]
        print(
            f"{path_name:<25} "
            f"{s['mean_total_s']:<12.3f} "
            f"±{s['std_total_s']:<8.3f} "
            f"{s['min_total_s']:<12.3f} "
            f"{s['mean_peak_vram_mb']:<15.0f} "
            f"{s['mean_steps_per_second']:<10.2f}"
        )

    # Speedup calculations
    print("-" * 90)
    if "baseline" in all_results and all_results["baseline"] is not None:
        base_time = all_results["baseline"]["stats"]["mean_total_s"]
        for path_name, data in all_results.items():
            if data is None or path_name == "baseline":
                continue
            path_time = data["stats"]["mean_total_s"]
            speedup = base_time / path_time
            vram_delta = data["stats"]["mean_peak_vram_mb"] - all_results["baseline"]["stats"]["mean_peak_vram_mb"]
            sign = "+" if vram_delta >= 0 else ""
            print(
                f"  {path_name:<23} {speedup:.2f}x speed   "
                f"VRAM: {sign}{vram_delta:.0f}MB "
                f"({'slower' if speedup < 1 else 'FASTER'}, "
                f"{'MORE' if vram_delta > 0 else 'less'} VRAM)"
            )

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Native diffusers benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="flux-schnell")
    parser.add_argument("--path", choices=list(LOADERS.keys()) + ["all"], default="all")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs to discard")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs")
    parser.add_argument("--save-output", action="store_true", help="Save first output PNG per path")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")
    parser.add_argument("--prompt", default=None, help="Override prompt")
    parser.add_argument("--steps", type=int, default=None, help="Override steps")
    parser.add_argument("--size", default=None, help="Override size (WxH, e.g. 768x768)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model].copy()
    if args.prompt:
        model_cfg["prompt"] = args.prompt
    if args.steps:
        model_cfg["default_steps"] = args.steps
    if args.size:
        w, h = map(int, args.size.lower().split("x"))
        model_cfg["default_size"] = (w, h)

    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU info
    print(f"\n{'=' * 60}")
    print(f"Phase 1 Benchmark: {args.model}")
    print(f"Prompt: {model_cfg['prompt'][:80]}...")
    print(f"Steps: {model_cfg['default_steps']}  Size: {model_cfg['default_size'][0]}x{model_cfg['default_size'][1]}")
    print(f"Warmup: {args.warmup}  Timed runs: {args.runs}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    import diffusers
    print(f"Diffusers: {diffusers.__version__}")
    print(f"{'=' * 60}\n")

    # Determine which paths to test
    paths = list(LOADERS.keys()) if args.path == "all" else [args.path]

    all_results = {}

    for path_name in paths:
        print(f"\n{'─' * 60}")
        print(f"Loading: {path_name}")
        print(f"{'─' * 60}")

        reset_vram()

        try:
            loader = LOADERS[path_name]
            pipe, description = loader(model_cfg)
            print(f"  {description}")
            print(f"  Load VRAM: {get_system_vram_mb()}MB system, "
                  f"{torch.cuda.memory_allocated()/1e6:.0f}MB allocated")
        except Exception as e:
            print(f"  ❌ LOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[path_name] = None
            continue

        print(f"\n  Running benchmark...")
        try:
            results = run_benchmark(
                pipe, model_cfg,
                num_warmup=args.warmup,
                num_timed=args.runs,
                save_output=args.save_output,
                output_dir=output_dir / path_name if args.save_output else None,
            )
            stats = compute_stats(results["timed"])
            all_results[path_name] = {
                "description": description,
                "warmup": results["warmup"],
                "timed": results["timed"],
                "stats": stats,
            }
            print(f"\n  ✅ Mean: {stats['mean_total_s']:.3f}s ± {stats['std_total_s']:.3f}s")
            print(f"     Peak VRAM: {stats['mean_peak_vram_mb']:.0f}MB")
            print(f"     Steps/s: {stats['mean_steps_per_second']:.2f}")
        except Exception as e:
            print(f"  ❌ BENCHMARK FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[path_name] = None

        # Cleanup
        del pipe
        reset_vram()
        time.sleep(5)  # cooldown between paths

    # Comparison report
    print_comparison(all_results)

    # Save JSON report
    report_path = output_dir / f"benchmark_{args.model}_{int(time.time())}.json"
    report = {
        "model": args.model,
        "model_cfg": {k: v for k, v in model_cfg.items() if k != "prompt"},
        "prompt": model_cfg["prompt"],
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "results": {},
    }
    import diffusers
    report["diffusers_version"] = diffusers.__version__

    for path_name, data in all_results.items():
        if data is None:
            report["results"][path_name] = {"status": "failed"}
        else:
            report["results"][path_name] = {
                "description": data["description"],
                "stats": data["stats"],
                "timed_runs": data["timed"],
                "warmup_runs": data["warmup"],
            }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
