"""Run Lance inference with GGUF quantized weights.

Loads model architecture via meta-init, dequantizes GGUF tensors to bf16,
and streams into the model's state dict.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def _meta_init():
    orig_empty = torch.empty

    def _empty_meta(*sizes, **kw):
        kw.setdefault("device", "meta")
        return orig_empty(*sizes, **kw)

    torch.empty = _empty_meta
    try:
        yield
    finally:
        torch.empty = orig_empty


def _dequant_tensor(tensor):
    """Dequantize a GGUF tensor to bf16 via gguf.dequantize."""
    import gguf
    ttype = tensor.tensor_type

    if ttype == gguf.GGMLQuantizationType.F32:
        return torch.from_numpy(tensor.data.copy()).to(torch.bfloat16)
    elif ttype == gguf.GGMLQuantizationType.F16:
        return torch.from_numpy(tensor.data.copy()).to(torch.bfloat16)

    # Block-quantized (Q5_K, Q6_K, etc.) — use gguf.dequantize
    deq = gguf.dequantize(tensor.data, ttype)
    return torch.from_numpy(deq).to(torch.bfloat16)


def _set_param_by_path(model, name: str, tensor: torch.Tensor):
    """Replace a parameter by dotted path (e.g. 'layers.0.weight')."""
    parts = name.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, torch.nn.ModuleDict):
            obj = obj[part]
        elif isinstance(obj, torch.nn.ModuleList):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    attr = parts[-1]
    if attr in obj._parameters:
        obj._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
    elif attr in obj._buffers:
        obj._buffers[attr] = tensor
    else:
        # May be a direct attribute
        setattr(obj, attr, torch.nn.Parameter(tensor, requires_grad=False))


def stream_gguf_weights(model, gguf_path: Path, device="cuda"):
    """Load GGUF weights into a meta-initialized model."""
    import gguf
    print(f"[gguf-stream] loading {gguf_path}")
    t0 = time.time()

    reader = gguf.GGUFReader(str(gguf_path))
    # Build name→shape map from state_dict for matching
    own_shapes = {n: p.shape for n, p in model.state_dict().items()}
    loaded = 0
    skipped = 0
    skipped_names = []

    for tensor in reader.tensors:
        name = tensor.name
        if name == "latent_pos_embed.pos_embed":
            skipped += 1
            continue
        if name not in own_shapes:
            skipped += 1
            if len(skipped_names) < 5:
                skipped_names.append(name)
            continue

        target_shape = own_shapes[name]
        weight = _dequant_tensor(tensor)

        # gguf.dequantize transposes some tensors — reshape to match
        if weight.shape != target_shape:
            if weight.numel() == target_shape.numel() if hasattr(target_shape, 'numel') else weight.numel() == np.prod(target_shape):
                weight = weight.reshape(tuple(target_shape))
            else:
                skipped += 1
                continue

        with torch.no_grad():
            weight_device = weight.to(device)
            _set_param_by_path(model, name, weight_device)
        loaded += 1
        del weight

        if loaded % 100 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  {loaded}/{len(reader.tensors)} loaded ({alloc:.1f}GB)", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[gguf-stream] {loaded} loaded, {skipped} skipped in {time.time()-t0:.1f}s")
    if skipped_names:
        print(f"[gguf-stream] first skipped: {skipped_names}")
    print(f"[gguf-stream] cuda mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Materialize any remaining meta tensors (buffers and params not in GGUF)
    meta_count = 0
    for name, param in model.named_parameters():
        if param.is_meta:
            with torch.no_grad():
                param.data = torch.zeros(param.shape, dtype=param.dtype, device=device)
            meta_count += 1
    for name, buf in model.named_buffers():
        if buf.is_meta:
            with torch.no_grad():
                buf.data = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
            meta_count += 1
    if meta_count:
        print(f"[gguf-stream] materialized {meta_count} remaining meta tensors")


def _patch_for_gguf(gguf_path: Path):
    """Patch inference_lance to use GGUF weights with meta-init.

    Flow:
    1. Model constructors use meta-init (no memory allocation for params)
    2. Lance.__init__ is patched to stream GGUF weights AFTER construction
       but BEFORE inference_lance.py calls model.to(DEVICE)
    3. GGUF weights are loaded directly to CUDA, replacing meta tensors
    4. model.to(DEVICE) succeeds because all params are now real tensors
    """
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    _OQwen = Qwen2ForCausalLM.__init__
    _OViT = Qwen2_5_VisionTransformerPretrainedModel.__init__
    _OLance = Lance.__init__

    def _Q(self, c):
        with _meta_init():
            _OQwen(self, c)

    def _V(self, c):
        with _meta_init():
            _OViT(self, c)

    _gguf_streamed = [False]

    def _L(self, *a, **k):
        with _meta_init():
            _OLance(self, *a, **k)
        # Stream GGUF weights immediately after Lance construction.
        # This replaces all meta tensors with real dequantized weights
        # on CUDA, before inference_lance.py calls model.to(DEVICE).
        if not _gguf_streamed[0]:
            _gguf_streamed[0] = True
            print("[gguf-patch] Streaming GGUF weights in Lance.__init__")
            stream_gguf_weights(self, gguf_path)
            if torch.cuda.is_available():
                print(f"[gguf-patch] cuda mem after stream: "
                      f"{torch.cuda.memory_allocated()/1e9:.2f} GB")

    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L

    def _gguf_loader(model, model_args):
        # init_from_model_path_if_needed is called at line 518, AFTER
        # to(DEVICE). GGUF weights already loaded in Lance.__init__,
        # so this is a no-op.
        print("[gguf-loader] weights already loaded, skipping")
        class _M:
            missing_keys: list[str] = []
            unexpected_keys: list[str] = []
        return _M()

    IL.init_from_model_path_if_needed = _gguf_loader
    print("[patch] meta-init + GGUF streaming in Lance.__init__ installed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--gguf_path", type=Path, required=True)
    ap.add_argument("--vit_path", default="downloads/Qwen2.5-VL-ViT")
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--save_path_gen", default=None)
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--video_height", type=int, default=768)
    ap.add_argument("--video_width", type=int, default=768)
    ap.add_argument("--validation_num_timesteps", type=int, default=30)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--example_json", default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    if args.save_path_gen is None:
        tag = args.gguf_path.stem
        args.save_path_gen = f"results/{tag}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.resolution is None:
        args.resolution = "image_768res" if args.task in ("t2i", "image_edit", "x2t_image") else "video_480p"

    sys.argv = [
        "inference_lance.py",
        "--model_path",            args.model_path,
        "--vit_path",              args.vit_path,
        "--vit_type",              "qwen_2_5_vl_original",
        "--llm_qk_norm",           "true",
        "--llm_qk_norm_und",       "true",
        "--llm_qk_norm_gen",       "true",
        "--tie_word_embeddings",   "false",
        "--validation_num_timesteps", str(args.validation_num_timesteps),
        "--validation_timestep_shift", "3.5",
        "--copy_init_moe",         "true",
        "--max_num_frames",        "121",
        "--max_latent_size",       "64",
        "--latent_patch_size",     "1", "1", "1",
        "--visual_und",            "true",
        "--visual_gen",            "true",
        "--vae_model_type",        "wan",
        "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template",   "false",
        "--cfg_type",              "0",
        "--validation_data_seed",  str(args.seed),
        "--video_height",          str(args.video_height),
        "--video_width",           str(args.video_width),
        "--num_frames",            str(args.num_frames),
        "--task",                  args.task,
        "--save_path_gen",         args.save_path_gen,
        "--resolution",            args.resolution,
        "--text_template",         "true",
        "--cfg_text_scale",        str(args.cfg_scale),
        "--use_KVcache",           "true",
    ]
    if args.example_json:
        sys.argv.extend(["--val_dataset_config_file", args.example_json])

    _patch_for_gguf(args.gguf_path.resolve())

    import inference_lance
    inference_lance.main()


if __name__ == "__main__":
    main()
