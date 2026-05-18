# HY-Motion — Original Workflow

**Source**: Vendor code at `vendor/hymotion/`
**Type**: Text-to-3D human motion generation
**Architecture**: HunyuanMotion MMDiT transformer with ODE sampling

## Inference Pipeline

### Stage 1: Text Encoding
```
text prompt → text_encoder (Qwen3-8B + CLIP)
  → vtxt_input (visual text features)
  → ctxt_input (context text features)
```

### Stage 2: ODE Sampling
```
random noise → motion_transformer (ODE integration, 50 Euler steps)
  → each step: cfg_scale guided forward pass through HunyuanMotionMMDiT
  → denoised motion latent
```

### Stage 3: Motion Decoding
```
motion latent → decode_motion_from_latent()
  → smoothing + body model
  → SMPL format body parameters (3D motion data)
```

## Components

| Module | Role | Size | Notes |
|--------|------|------|-------|
| motion_transformer | HunyuanMotionMMDiT — core denoising | ~large | ODE-based flow matching |
| text_encoder | Qwen3-8B + CLIP for text understanding | ~16GB | Two sub-models combined |
| MLP / MLPEncoder | Auxiliary projections | small | Conditioning projections |
| TimestepEmbeddingEncoder | Time step encoding | small | ODE timestep embedding |
| RotaryEmbedding | Positional encoding | small | Attention RoPE |

## Key Characteristics

- **ODE-based sampling** (not standard diffusion) — uses `torchdiffeq.odeint` with Euler integration
- **Qwen3-8B text encoder** — same backbone as MOSS, but used for text understanding not generation
- **50 ODE steps** — motion_transformer called 50 times in sequence
- **SMPL body model output** — standard 3D human motion representation
- **Workspace setup** — creates temp directory with symlinks to Qwen3 + CLIP weights + stats files

## Relationship to Other Models

- **Qwen3-8B** text encoder is the same family as MOSS's backbone (but used differently — encoding, not generation)
- **motion_transformer** is a Hunyuan-family MMDiT (related to HunyuanVideo architecture)
