import torch, json, os, gc, time, shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/mnt/data/models/video/wan2.1-vace-14b-fp8-diffusers/transformer")
DST = Path("/mnt/data/models/video/wan2.1-vace-14b-fp8-scaled")
DST.mkdir(parents=True, exist_ok=True)
(DST / "transformer").mkdir(parents=True, exist_ok=True)

shards = sorted(SRC.glob("*.safetensors"))
print(f"Processing {len(shards)} FP8 shards — adding weight_scale tensors...", flush=True)

for shard in shards:
    t0 = time.perf_counter()
    out_dict = {}
    scale_count = 0
    
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.float8_e4m3fn:
                max_abs = tensor.to(torch.float32).abs().amax()
                scale = (max_abs / 448.0).clamp(min=1e-12)
                out_dict[key] = tensor
                out_dict[key.replace(".weight", ".weight_scale")] = scale
                scale_count += 1
            else:
                out_dict[key] = tensor
    
    out_path = DST / "transformer" / shard.name
    save_file(out_dict, str(out_path))
    elapsed = time.perf_counter() - t0
    print(f"  {shard.name}: {scale_count} scales ({elapsed:.1f}s)", flush=True)
    del out_dict; gc.collect()

# Copy index + add scale entries
idx_path = SRC / "diffusion_pytorch_model.safetensors.index.json"
with open(idx_path) as f:
    idx = json.load(f)
new_map = {}
for k, v in idx["weight_map"].items():
    new_map[k] = v
    if k.endswith(".weight") and not k.endswith("._scale"):
        new_map[k.replace(".weight", ".weight_scale")] = v
idx["weight_map"] = new_map
with open(DST / "transformer" / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
    json.dump(idx, f)

# Config with compressed-tensors FP8
cfg_src = SRC / "config.json"
with open(cfg_src) as f:
    cfg = json.load(f)
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
        "weights": {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
        "targets": ["Linear"]
    }},
    "ignore": ["condition_embedder*", "norm_out*", "patch_embedding*", "proj_out*", "scale_shift_table*"]
}
with open(DST / "transformer" / "config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# Copy remaining model files
bf16 = Path("/mnt/data/models/video/wan2.1-vace-14b-diffusers")
for item in bf16.iterdir():
    if item.name == "transformer": continue
    dst = DST / item.name
    if dst.exists(): continue
    if item.is_dir(): shutil.copytree(item, dst)
    else: shutil.copy2(item, dst)

trans_sz = sum(f.stat().st_size for f in (DST/"transformer").glob("*.safetensors")) / 1e9
te_sz = sum(f.stat().st_size for f in (DST/"text_encoder").glob("*.safetensors")) / 1e9
print(f"\n✅ FP8 SCALED MODEL READY", flush=True)
print(f"  Transformer: {trans_sz:.1f}GB  Text encoder: {te_sz:.1f}GB", flush=True)
print(f"  Path: {DST}", flush=True)
