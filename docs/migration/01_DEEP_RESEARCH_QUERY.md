# Deep Research Query — Send to Deep Research Agent

## Instructions for the deep research agent

The following is a comprehensive research query about migrating a video/image
generation platform from Wan2GP/mmGP (a custom VRAM management library) to
native HuggingFace diffusers APIs and SGLang Diffusion.

We have already verified that `enable_group_offload(use_stream=True)` and
`enable_layerwise_casting(storage_dtype=fp8)` exist in diffusers 0.37.0. We
need you to verify the PERFORMANCE claims, find real-world benchmarks, and
identify any gotchas we haven't considered.

Our hardware: NVIDIA RTX 4090 (24GB VRAM), 59GB RAM, Ray Serve on Kubernetes.

---

## THE QUERY

I am migrating an AI media generation platform away from Wan2GP and its VRAM
management library mmGP (Memory Management for the GPU Poor, by DeepBeepMeep)
to native HuggingFace diffusers APIs and optionally SGLang Diffusion.

Our hardware is an NVIDIA RTX 4090 with 24GB VRAM and 59GB system RAM, running
on Kubernetes with Ray Serve for orchestration. We generate images and video
using models including Wan 2.1/2.2 (14B), FLUX (schnell, dev, chroma), Qwen-
Image, Z-Image, LTX-Video/LTX2/LTX-2.3, Anima, TRELLIS (3D), and various TTS
models (Kokoro, Moss, Index TTS).

I need comprehensive research on the following topics. For each, please provide
specific performance numbers, code examples, benchmark results, and any known
issues or limitations.

### 1. diffusers `enable_group_offload` performance benchmarks

We've verified that `pipeline.transformer.enable_group_offload(onload_device=
cuda, offload_device=cpu, offload_type="block_level", use_stream=True)` exists
in diffusers 0.37.0. We need to know:

a) How does generation speed with `enable_group_offload(use_stream=True)`
   compare to having the model fully resident in VRAM (no offloading)? What is
   the typical overhead percentage for block-level group offloading with async
   CUDA streams?

b) How does it compare to mmGP's block-level streaming? Has anyone published
   head-to-head benchmarks between diffusers group_offload and mmGP on consumer
   GPUs (RTX 3090/4090)?

c) What is the actual mechanism? Does `use_stream=True` create a dedicated CUDA
   stream for H2D transfers and overlap them with compute? How many blocks can
   be prefetched ahead? Is there a `num_blocks_per_group` tuning guide?

d) Which diffusers model architectures officially support group offloading?
   Does it work with FluxTransformer2DModel, WanTransformer3DModel,
   LTXVideoTransformer3D, CosmosTransformer3DModel, AutoencoderKLQwenImage? Is
   there a `_supports_group_offloading` flag, and which models have it set?

e) Can `enable_group_offload` be combined with `enable_layerwise_casting`
   (FP8 storage + bf16 compute) simultaneously? Are there any conflicts or
   performance interactions between block offloading and layerwise weight
   casting?

f) Does group offloading work correctly with LoRA adapters loaded via PEFT
   (`pipe.load_lora_weights`)? Does the stream prefetch interfere with adapter
   weight application? Are there known bugs?

g) Does group offloading work with `torch.compile`? Are there compilation
   edge cases or graph breaks?

### 2. SGLang Diffusion — production readiness and performance

SGLang Diffusion (from LMSYS/UC Berkeley/Stanford) was recently announced as
a high-performance inference framework for image and video diffusion models.
We need to know:

a) What is the actual performance of SGLang Diffusion on consumer GPUs (RTX
   4090, 24GB VRAM)? The launch claims "up to 5.9× faster inference" — what is
   the baseline, and what are the real-world numbers for Wan, FLUX, Qwen-Image,
   and LTX-Video specifically?

b) How does SGLang's sleep/wake mechanism work for VRAM scale-to-zero? When a
   model is "asleep," what is the VRAM footprint (just CUDA context, or zero)?
   What is the wake latency to bring a model back from sleep? Is it using
   pinned memory for fast PCIe transfer?

c) SGLang supports `--ltx2-two-stage-device-mode` with modes `resident`,
   `snapshot`, and `original`. What exactly does each mode do? What are the
   VRAM/speed tradeoffs for each? Which mode is recommended for a 24GB GPU?

d) Can SGLang Diffusion be deployed alongside native diffusers pipelines in the
   same Ray Serve cluster? Are there library version conflicts between sglang
   and diffusers/transformers? Can they coexist in the same Python environment,
   or do they need separate containers?

e) What models does SGLang Diffusion officially support as of June 2026? Does
   it support Anima, TRELLIS, or any TTS models? What is the process for adding
   support for a new model architecture?

f) How stable is SGLang Diffusion for production use? Are there known crashes,
   memory leaks, or output quality issues? How does it handle concurrent
   requests and batching for diffusion models?

g) What is the OpenAI-compatible API surface? Can it handle image-to-video,
   video-to-video, and multi-stage generation (e.g., LTX two-stage with
   upscaler)?

### 3. Native diffusers pipeline features for advanced generation

We need to verify that native diffusers pipelines support the advanced features
we currently use through Wan2GP handlers:

a) LTX-Video pipeline: Does `LTXVideoPipeline` in diffusers support first-frame
   and last-frame conditioning natively? What parameters control this
   (`media_frame_number`, `strength`, `num_prefix_latent_frames`,
   `prefix_latents_mode`)? Can we do image-to-video, video-to-video, and
   looping generation through the standard pipeline API?

b) Wan 2.1 pipeline: Does `WanPipeline` / `WanImageToVideoPipeline` support
   first-frame/last-frame conditioning? What about the FLF2V (first-last-frame-
   to-video) variant? Are these exposed as pipeline arguments?

c) Multi-stage generation: Can we run a model's pipeline in two stages (low-
   resolution denoising → high-resolution refinement) through native diffusers?
   Does LTX-Video's two-stage generation work natively, or does it require
   custom code?

d) Custom latent injection: Can we pass pre-generated latents to a diffusers
   pipeline via `pipe(prompt, latents=custom_latents)`? Which pipelines support
   this? Does it work correctly with group offloading?

e) Prompt relay / Director mode: We currently use temporal prompt segments
   (different prompts for different frame ranges). Is there native diffusers
   support for this, or does it require custom scheduler/attention code?

f) LoRA hot-swapping: With PEFT, can we load/unload/swap LoRA adapters between
   generation requests without reloading the base model? What is the swap
   latency? Can multiple LoRAs be active simultaneously with independent
   weights?

g) VAE tiling and slicing: Does `vae.enable_tiling()` and
   `vae.enable_slicing()` work correctly with Qwen-Image VAE
   (AutoencoderKLQwenImage)? What tile sizes are recommended for 24GB VRAM?

### 4. transformers v5 migration risks

a) What are ALL the breaking changes in transformers v5.0 that affect diffusion
   model pipelines? Specifically: tokenizer changes (decode API returning lists,
   apply_chat_template returning BatchEncoding, additional_special_tokens rename),
   encode_plus removal, config file format changes.

b) Is diffusers 0.37.0 compatible with transformers 5.x? Or does upgrading
   transformers to 5.x require also upgrading diffusers? Are there known
   incompatibilities?

c) Does optimum-quanto (used for quantization) work with transformers 5.x?
   Is there a compatible version?

d) What is the migration path for a project currently on transformers 4.57.3?
   What code changes are required? Is there an automated migration tool?

e) For our use case (diffusion models, not LLMs), do the transformers v5
   breaking changes even affect us? Or are the breaking changes primarily in
   NLP/tokenizer code paths that diffusion models don't use?

### 5. Ray Serve integration patterns

a) What is the recommended pattern for deploying multiple diffusers pipelines
   in Ray Serve with per-model scale-to-zero? How do you configure
   `min_replicas=0` for GPU-bound inference deployments?

b) When Ray Serve kills a deployment replica (scale-to-zero), does it properly
   release CUDA memory? Are there known issues with CUDA memory not being freed
   on deployment deletion?

c) Can different Ray Serve deployments use different versions of diffusers/
   transformers (via separate virtual environments or containers)? Or do all
   deployments in a Ray cluster share the same Python environment?

d) What is the recommended way to pass generated media (images, video) between
   Ray Serve deployments? Base64 encoding via HTTP, or Ray's Plasma object store?
   What are the size limits and performance characteristics?

e) How do you handle GPU allocation in Ray Serve when multiple deployments
   need GPU access? Does Ray's GPU scheduling work correctly with Kubernetes?

### 6. Model-specific questions

a) Anima (circlestone-labs/Anima): Is this model available in diffusers as a
   standard pipeline? If not, what is the minimal code needed to run it with
   native diffusers components (CosmosTransformer3DModel + Qwen3 text encoder +
   T5 tokenizer + LLM adapter + Qwen-Image VAE)?

b) LTX-2.3 (Lightricks/LTX-2.3): What is the exact diffusers pipeline class for
   this model? Does it support the two-stage generation natively? What VRAM is
   needed with group offloading + layerwise casting?

c) Wan 2.2 Animate 14B: Is there a diffusers pipeline for this? Does
   `enable_group_offload` work with the Wan transformer architecture? What is
   the minimum VRAM with group offloading?

d) TRELLIS 3D: Is there a diffusers integration for TRELLIS? If not, what is
   the cleanest way to run it — direct from the Microsoft repo, or is there a
   community wrapper?

e) Z-Image: Is there a diffusers pipeline for Z-Image? Does it support the
   Turbo (4-step) variant? What about LoRA loading?

### 7. Performance comparison methodology

a) What is the recommended way to benchmark diffusion model inference to get
   reproducible results? How do you account for CUDA warmup, memory
   fragmentation, and thermal throttling?

b) What VRAM metrics should we track? `torch.cuda.max_memory_allocated()` vs
   `torch.cuda.memory_reserved()` vs nvidia-smi reported usage?

c) For video generation specifically, how do you benchmark fairly across
   different offloading strategies? Frame generation time, total generation
   time, or time-to-first-frame?

---

## What we've already verified (don't re-research these)

- ✅ `enable_group_offload` exists in diffusers 0.37.0 (tested on our worker)
- ✅ `enable_layerwise_casting` exists in diffusers 0.37.0 (tested on our worker)
- ✅ SGLang Diffusion exists at docs.sglang.io/docs/sglang-diffusion
- ✅ DiffSynth-Studio exists at diffsynth-studio-doc.readthedocs.io
- ✅ mmGP is model-agnostic (no model-specific code in the 6,361-line library)
- ✅ Wan2GP's handler layer (261k lines) is where model-support lag lives
- ✅ `use_async_weight_loading` parameter does NOT exist (AI hallucination)
- ✅ transformers v5 has breaking changes (migration guide exists)
- ✅ Our hardware: RTX 4090, 24GB VRAM, 59GB RAM
- ✅ Our software: PyTorch 2.10, transformers 4.57.3, diffusers 0.37.0, mmGP 3.7.6

## What we most need answered

Priority 1: Is `enable_group_offload(use_stream=True)` fast enough to replace
mmGP on a 4090? (This is the make-or-break question for the entire migration.)

Priority 2: Is SGLang Diffusion production-ready enough to rely on for standard
models (Wan, FLUX, Qwen-Image, LTX)?

Priority 3: Are there hidden gotchas in combining group_offload + layerwise_
casting + PEFT LoRAs that we haven't considered?
