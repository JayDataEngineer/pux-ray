# Model Orchestration Spec

## What This Is

A specification for describing, triaging, and optimizing GPU models under one orchestration system on a single RTX 4090 (24GB VRAM).

This spec SUPERCLASSES Wan2GP's handler pattern. Wan2GP's handler says "give me a pipe dict and a generate function." This spec says what the model IS, what optimizations are available, what's currently used, and what the gap is.

## Integration Levels

### Built-in
Wan2GP ships the handler AND manages the entire inference flow. All optimizations apply automatically.

- Wan 2.1, HunyuanVideo, Flux, LTX Video, Kandinsky 5, HiDream, SkyReels, Vace, etc.
- We just call the handler. Zero custom code.
- Full Wan2GP optimization stack: nanovllm, quantization, CUDA graphs, mmgp, attention backends, schedulers.

### Native Custom
Our handler code, but the full inference flow runs through Wan2GP's optimization stack. Wan2GP manages the pipeline — you can't tell it apart from a built-in.

Requirements:
- Model flow maps to Wan2GP's standard patterns (diffusion loop, autoregressive decode, etc.)
- All nn.Modules decomposed into pipe dict
- Wan2GP's shared layer handles: mmgp, quantization, attention, scheduling

Example: faster_qwen3_tts uses nanovllm + CUDA graphs + mmgp through Wan2GP's native path. It IS native.

### Partial
We use Wan2GP primitives a la carte. The pipeline flow is ours — Wan2GP cannot manage the full inference because the model doesn't fit a standard pattern.

- Wan2GP provides: mmgp weight swapping, quantization backends, specific utilities
- We provide: pipeline orchestration, stage management, custom VRAM management, intermediate handling
- Wan2GP CANNOT: prefetch across stages, schedule components, manage intermediate tensors

This is not a failure. Multi-stage pipelines (TRELLIS, AniGen) with custom VRAM management simply don't map to Wan2GP's single-generate pattern. The spec honestly documents what we get and what we don't.

## Component Descriptor

Each model breaks down into components. For each component:

```
name:           string          — identifier
type:           string          — transformer | encoder | decoder | vae | flow_model | custom
size_gb:        float           — VRAM footprint at full precision
precision:      string          — bf16 | fp16 | fp32 | mixed
quantizable:    list[string]    — [int8, int4, gguf_q4, fp8, nvfp4] — what Wan2GP quant backends can target this
shared:         string | null   — component group name (e.g., "dinov2-vitl") for cross-model sharing
replaceable:    string | null   — drop-in alternative (e.g., "birefnet-lite")
exclude_mmgp:   bool            — keep outside mmgp pipe (FP32 helpers, lightweight components)
wan2gp_native:  bool            — can Wan2GP manage this component's full lifecycle?
```

## Pipeline Stages

```
name:           string          — stage identifier
components:     list[string]    — which components are active
depends_on:     list[string]    — previous stages that must complete first
gpu:            bool            — needs GPU?
type:           string          — encode | denoise | decode | postprocess | export
repeated:       bool            — called N times in a loop (diffusion steps, AR decode)
batchable:      bool            — can this stage process multiple inputs simultaneously?
```

## Optimization Taxonomy

| Optimization | Source | What it does | Applies to |
|-------------|--------|-------------|------------|
| mmgp weight swapping | Wan2GP shared | Stream weights between CPU/GPU on forward() | All GPU components |
| INT8 quantization (JIT) | Wan2GP qtypes + mmgp | Quantize to INT8 during loading via optimum.quanto | Transformers, linear layers |
| GGUF quantization | Wan2GP qtypes + llama.cpp | Pre-quantized GGUF with CUDA kernels | LLM backbones, autoregressive models |
| NVFP4 quantization | Wan2GP qtypes | NVIDIA FP4 format | Large transformers |
| Pre-made quants (unsloth, etc.) | External | Download pre-quantized weights, skip JIT quant | Any model with available quants |
| nanovllm (paged attention) | Wan2GP shared | vLLM-style paged KV cache + scheduling | Autoregressive models (LLM, TTS) |
| CUDA graphs | Wan2GP shared | Capture fixed-shape inference as static graph | Autoregressive decode steps |
| MagCache (step skipping) | Wan2GP shared | Skip diffusion steps below threshold | Diffusion models with repeated transformer calls |
| Text encoder cache | Wan2GP shared | LRU cache for text embeddings on CPU | Models with expensive text encoding |
| SageAttention 2 | Wan2GP shared | Flash attention alternative | Transformers with compatible attention |
| Component sharing | Orchestration layer | One copy of shared components (e.g., DINOv2) | Models with common components |
| Drop-in replacements | Orchestration layer | Swap heavy components for lighter alternatives | Background removal, normalization, etc. |
| Batch inference | Wan2GP native | Process N samples through same model simultaneously | Diffusion stages, flow matching stages |

## Triage Process

For each model, answer per component:

1. **Can Wan2GP quantize it?** → Which backends? What precision loss?
2. **Does another model already load it?** → Shared component group.
3. **Is there a lighter replacement?** → Drop-in alternative.
4. **Can Wan2GP manage its full lifecycle?** → Native vs Partial.
5. **Can this stage batch?** → Multiple inputs through same weights.

The triage produces the model's OPTIMIZATIONS.md — the gap between what Wan2GP CAN do and what we're CURRENTLY using.
