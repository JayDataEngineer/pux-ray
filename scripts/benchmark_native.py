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
        "repo": "/models/flux-schnell",  # local persistent path (no re-download)
        "repo_fallback": "black-forest-labs/FLUX.1-schnell",  # HF fallback
        "pipeline_cls": "FluxPipeline",
        "default_steps": 4,
        "default_guidance": 0.0,
        "default_size": (1024, 1024),
        "prompt": "a cinematic photo of a golden retriever puppy in a field of wildflowers, golden hour lighting, shallow depth of field",
        "license": "Apache-2.0",
    },
    "flux-dev": {
        "repo": "/models/flux-dev",  # local persistent path
        "repo_fallback": "black-forest-labs/FLUX.1-dev",
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


def load_bf16_resident(model_cfg):
    """A1: BF16 fully resident — gold standard for speed and quality."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    return pipe, "bf16 resident (pipe.to cuda, all components on GPU)"


def load_bf16_group_offload(model_cfg):
    """A3/B1: BF16 group_offload — streaming without quantization.
    Text encoders + VAE resident, transformer blocks stream in BF16."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    # Text encoders + VAE resident on GPU
    pipe.text_encoder.to("cuda")
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()
    # Group offload WITHOUT layerwise_casting — pure BF16 streaming
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        record_stream=True,
    )
    return pipe, "bf16 group_offload (stream, NO quantization)"


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


def load_bf16_sequential(model_cfg):
    """VRAM technique: sequential_cpu_offload — layer-by-layer (slowest, lowest VRAM)."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe, "bf16 sequential_cpu_offload (layer-by-layer)"


def load_fp8_mixed_cpu_offload(model_cfg):
    """Mixed precision: text encoders FP8, transformer BF16, model_cpu_offload.
    This tests the 'quantize where it's invisible' approach."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    # Quantize ONLY text encoders to FP8 (quality impact invisible)
    try:
        pipe.text_encoder.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        te1_status = "text_encoder FP8"
    except Exception as e:
        te1_status = f"text_encoder FP8 FAILED ({e})"

    try:
        pipe.text_encoder_2.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        te2_status = "text_encoder_2 FP8"
    except Exception as e:
        te2_status = f"text_encoder_2 FP8 FAILED ({e})"

    # Transformer stays BF16 (precision-critical)
    pipe.enable_model_cpu_offload()
    return pipe, f"mixed: {te1_status} + {te2_status} + transformer BF16 + model_cpu_offload"


def load_bf16_compile_cache(model_cfg):
    """Path A full: model_cpu_offload + compile_repeated_blocks + cache_accel.
    NOTE: compile + cache were found incompatible. This tests which one wins."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_cfg["repo"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Try compile first (cache is known incompatible with compile)
    try:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        opt_status = "compile_repeated_blocks: ON"
    except Exception as e:
        opt_status = f"compile FAILED ({e})"

    return pipe, f"Path A: model_cpu_offload + {opt_status}"


LOADERS = {
    "bf16-resident": load_bf16_resident,
    "bf16-cpu-offload": load_baseline,
    "bf16-sequential": load_bf16_sequential,
    "bf16-group-offload": load_bf16_group_offload,
    "fp8-group-offload": load_group_offload,
    "fp8-mixed-cpu-offload": load_fp8_mixed_cpu_offload,
    "compile-only": load_compile_only,
    "cache-only": load_cache_only,
    "compile-cache": load_bf16_compile_cache,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(pipe, model_cfg, num_warmup=3, num_timed=5, save_output=False, output_dir=None, config_name=""):
    """Run benchmark with warmup + timed runs. Returns metrics dict."""

    prompt = model_cfg["prompt"]
    steps = model_cfg["default_steps"]
    guidance = model_cfg["default_guidance"]
    width, height = model_cfg["default_size"]
    seed = 42

    results = {
        "warmup": [],
        "timed": [],
        "cold_start_s": None,
    }

    # === WARMUP (first run is cold start) ===
    for i in range(num_warmup):
        label = "COLD" if i == 0 else f"warmup {i+1}/{num_warmup}"
        print(f"  [{label}]", end="", flush=True)

        generator = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        _ = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.2f}s")
        results["warmup"].append(elapsed)
        if i == 0:
            results["cold_start_s"] = elapsed
        reset_vram()

    # === TIMED RUNS ===
    for i in range(num_timed):
        reset_vram()
        vram_before = get_system_vram_mb()

        print(f"  [timed {i+1}/{num_timed}]", end="", flush=True)

        generator = torch.Generator().manual_seed(seed)
        t_total_start = time.perf_counter()

        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
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

        # Save first timed output for quality comparison
        if save_output and i == 0 and output_dir:
            img = output.images[0]
            safe_name = config_name.replace("/", "_")
            outpath = output_dir / f"{safe_name}_seed{seed}.png"
            img.save(outpath)
            print(f"  Saved: {outpath}")

        # Cooldown between runs
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
    print("\n" + "=" * 105)
    print("BENCHMARK RESULTS — FAIR COMPARISON")
    print("=" * 105)
    print(f"{'Path':<25} {'Cold(s)':<10} {'Warm(s)':<10} {'Std':<8} {'Min(s)':<10} {'VRAM(MB)':<12} {'Steps/s':<8}")
    print("-" * 105)

    for path_name, data in all_results.items():
        if data is None:
            print(f"{path_name:<25} {'FAILED':<20}")
            continue
        s = data["stats"]
        cold = data.get("cold_start_s", 0) or 0
        print(
            f"{path_name:<25} "
            f"{cold:<10.2f} "
            f"{s['mean_total_s']:<10.3f} "
            f"±{s['std_total_s']:<6.3f} "
            f"{s['min_total_s']:<10.3f} "
            f"{s['mean_peak_vram_mb']:<12.0f} "
            f"{s['mean_steps_per_second']:<8.2f}"
        )

    # Find fastest for reference
    print("-" * 105)
    valid = {k: v for k, v in all_results.items() if v is not None}
    if valid:
        fastest = min(valid.items(), key=lambda x: x[1]["stats"]["mean_total_s"])
        print(f"  Fastest: {fastest[0]} at {fastest[1]['stats']['mean_total_s']:.3f}s")

    print("=" * 105)


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

    # Resolve model path: try local first, fall back to HF
    repo = model_cfg.get("repo", "")
    if not os.path.exists(repo) and "repo_fallback" in model_cfg:
        print(f"Local path '{repo}' not found, using HF: {model_cfg['repo_fallback']}")
        model_cfg["repo"] = model_cfg["repo_fallback"]

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
                save_output=True,  # always save first output
                output_dir=output_dir,
                config_name=path_name,
            )
            stats = compute_stats(results["timed"])
            all_results[path_name] = {
                "description": description,
                "warmup": results["warmup"],
                "cold_start_s": results.get("cold_start_s"),
                "timed": results["timed"],
                "stats": stats,
            }
            cold = results.get("cold_start_s", 0)
            print(f"\n  ✅ Cold: {cold:.2f}s  Warm: {stats['mean_total_s']:.3f}s ± {stats['std_total_s']:.3f}s")
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
