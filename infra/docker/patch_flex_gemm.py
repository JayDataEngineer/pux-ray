"""Patch flex_gemm source to work with triton 3.x and no-GPU build contexts.

Two problems with stock flex_gemm:
1. config.py calls torch.cuda.get_device_name() at import → crashes without GPU
2. Kernel files use @triton.jit which crashes with triton 3.x's new source-parsing

EXPLICIT_GEMM algorithm bypasses all triton kernels, so we replace the broken
modules with stubs. The real CUDA-backed ops in ops/spconv/ still work fine.

Usage: python patch_flex_gemm.py /path/to/flex_gemm_source_root
"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "."

triton_spconv = os.path.join(root, "flex_gemm/kernels/triton/spconv")
triton_grid = os.path.join(root, "flex_gemm/kernels/triton/grid_sample")

def stub_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# --- spconv stubs ---
stub_file(os.path.join(triton_spconv, "config.py"),
          "autotune_config = []\nallow_tf32 = True\nUSE_ON_THE_FLY_WEIGHT_TRANSPOSE = True\n")

for name in [
    "sparse_submanifold_conv_fwd_implicit_gemm",
    "sparse_submanifold_conv_bwd_implicit_gemm",
    "sparse_submanifold_conv_fwd_implicit_gemm_splitk",
    "sparse_submanifold_conv_bwd_implicit_gemm_splitk",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm",
    "sparse_submanifold_conv_bwd_masked_implicit_gemm",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk",
    "sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk",
    "sparse_conv_implicit_gemm",
    "sparse_conv_implicit_gemm_splitk",
    "sparse_conv_masked_implicit_gemm",
    "sparse_conv_masked_implicit_gemm_splitk",
]:
    fname = name.split(".")[-1] if "." in name else name
    stub_file(os.path.join(triton_spconv, f"{name}.py"),
              f"def {fname}(*a, **kw):\n    raise NotImplementedError('triton kernel unavailable — use EXPLICIT_GEMM')\n")

stub_file(os.path.join(triton_spconv, "__init__.py"),
          "from .config import autotune_config, allow_tf32, USE_ON_THE_FLY_WEIGHT_TRANSPOSE\n")

# --- grid_sample stubs ---
stub_file(os.path.join(triton_grid, "config.py"),
          "autotune_config = []\n")

# indice_weighed_sum_fwd — pure PyTorch fallback (replaces broken @triton.jit kernel)
# Computes weighted sum: out[m,c] = sum_v feats[indices[m,v], c] * weight[m,v]
stub_file(os.path.join(triton_grid, "indice_weighed_sum_fwd.py"), """\
import torch

def indice_weighed_sum_fwd(
    feats: torch.Tensor,       # [N, C]
    indices: torch.Tensor,     # [M, V] — may contain 0xffffffff for invalid
    weight: torch.Tensor,      # [M, V]
) -> torch.Tensor:
    M, V = indices.shape
    C = feats.shape[1]
    idx = indices.long()                                              # [M, V]
    # Mask invalid indices (0xffffffff for uint32 = -1 for int64)
    valid = (idx >= 0) & (idx < feats.shape[0])                       # [M, V]
    idx = idx.clamp(0, feats.shape[0] - 1)                            # clamp to valid range
    gathered = feats[idx]                                             # [M, V, C]
    w = weight.to(feats.dtype).unsqueeze(-1)                          # [M, V, 1] — match feats dtype
    weighted = gathered * w * valid.unsqueeze(-1)                     # [M, V, C]
    return weighted.sum(dim=1)                                        # [M, C]
""")

# indice_weighed_sum_bwd_input — pure PyTorch scatter fallback
stub_file(os.path.join(triton_grid, "indice_weighed_sum_bwd_input.py"), """\
import torch

def indice_weighed_sum_bwd_input(
    grad_output: torch.Tensor,  # [M, C]
    indices: torch.Tensor,      # [M, V]
    weight: torch.Tensor,       # [M, V]
    N: int,                     # number of input features
) -> torch.Tensor:
    M, V = indices.shape
    C = grad_output.shape[1]
    grad_input = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
    idx = indices.long()                                               # [M, V]
    valid = (idx >= 0) & (idx < N)                                    # [M, V]
    # contribution per neighbor
    contrib = grad_output.unsqueeze(1) * weight.unsqueeze(-1)          # [M, 1, C] * [M, V, 1] -> [M, V, C]
    for v in range(V):
        valid_v = valid[:, v]
        if valid_v.any():
            grad_input.index_add_(0, idx[valid_v, v], contrib[valid_v, v])
    return grad_input
""")

# indice_weighed_sum_bwd — stub (only needed by autograd, not called directly by ops)
stub_file(os.path.join(triton_grid, "indice_weighed_sum_bwd.py"),
          "def indice_weighed_sum_bwd(*a, **kw):\n    raise NotImplementedError('triton kernel unavailable')\n")

stub_file(os.path.join(triton_grid, "__init__.py"),
          "from .config import autotune_config\n"
          "from .indice_weighed_sum_fwd import indice_weighed_sum_fwd\n"
          "from .indice_weighed_sum_bwd_input import indice_weighed_sum_bwd_input\n"
          "from .indice_weighed_sum_bwd import indice_weighed_sum_bwd\n")

print("flex_gemm source patched (triton kernels stubbed)")
