# Orchestration Goals

## What We're Building

A single system that manages ALL GPU models on one RTX 4090 (24GB VRAM). Not ComfyUI (subprocess). Not llama.cpp (subprocess). The models and pipelines that run through Wan2GP's inference stack.

The orchestrator does not replace Wan2GP. It TRIAGES each model and wires it into Wan2GP's optimization stack as fully as possible.

## The Three Questions

For every model:

1. **What can Wan2GP optimize?** — Which of its shared primitives (nanovllm, quantization, CUDA graphs, mmgp, attention backends) actually apply?
2. **What can we share or swap?** — Common components across models (DINOv2), lighter drop-in replacements (BiRefNet-lite for rembg).
3. **What can't we touch?** — Custom pipeline stages, exotic components, models that don't map to Wan2GP's patterns. Leave these alone.

## Goal 1: Close the Optimization Gap

Most custom models currently use ONLY mmgp weight swapping from Wan2GP's entire stack. The gap between "what Wan2GP can do" and "what we use" is the optimization opportunity.

- MOSS (8B Qwen3) → could use nanovllm + CUDA graphs + GGUF quantization. Currently uses mmgp only.
- TRELLIS flow models → could use INT8 quantization. Currently BF16 only.
- AniGen → same as TRELLIS, plus shared DINOv2 with TRELLIS.
- faster_qwen3_tts → ALREADY native. Uses nanovllm + CUDA graphs + mmgp. This is the target for other models.

## Goal 2: Share Components Across Models

Models that load the same thing should share one copy.

- DINOv2 image conditioning → used by TRELLIS, AniGen, potentially others.
- Text encoders (T5, CLIP) → used by multiple diffusion models.
- Background removal → used by TRELLIS, AniGen, possibly swap for lighter model.

When switching from TRELLIS to AniGen, if DINOv2 is already in RAM, skip loading it.

## Goal 3: Quantize Heavy Components

Wan2GP has 5 quantization backends (GGUF, NVFP4, FP4, INT4, FP8). Most custom models use NONE of them.

Two paths:
- **JIT quantization**: Wan2GP quantizes during loading via optimum.quanto. Already available for any pipe dict component.
- **Pre-made quants**: Download INT8/GGUF weights from sources like unsloth. Skip JIT overhead, potentially better quality.

Target: TRELLIS flow models at INT8 could drop from ~10GB to ~5GB. That's the difference between "barely fits" and "has headroom."

## Goal 4: Enable Batch Where Possible

For diffusion/flow matching stages, batch_size > 1 amortizes weight loading across multiple samples.

- Flow matching stages in TRELLIS/AniGen: batch the denoising loop, run 2 images through same transformer call.
- The decode/export stages are sequential (varying-size sparse tensors) — don't batch those.

Not about running two full models simultaneously. About making each model pass faster.

## Goal 5: Honest Triage

Not everything can be optimized. Multi-stage pipelines with custom VRAM management (TRELLIS, AniGen) may always be Partial integration. That's fine. The spec documents it honestly.

The orchestrator's value is:
- Making Native models trivially easy to add (Wan2GP does everything)
- Closing the gap on Partial models (quantize what we can, share what we can, leave the rest alone)
- Never pretending a Partial model is Native

## What's Out of Scope

- ComfyUI — subprocess with its own runtime. Different system entirely.
- llama.cpp — subprocess with its own runtime. Different system entirely.
- CPU services (Kokoro, eSpeak, Faster-Whisper) — no GPU, no Wan2GP stack needed.
- Building new optimization primitives — Wan2GP's shared layer already has them.
- DAG scheduling / AOT compilation — the models already have fixed pipelines. We declare them, we don't discover them.

## Success Metric

When a new model is added, the triage should be:
1. Read the ORIGINAL-WORKFLOW.md (how does it work?)
2. Answer the three questions (what can Wan2GP optimize, share, or leave alone?)
3. Write the OPTIMIZATIONS.md (the triage result)
4. Implement the handler at the appropriate integration level
5. The gap between "available optimizations" and "applied optimizations" should be documented and shrinking over time
