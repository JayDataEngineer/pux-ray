"""Patches SageAttention setup.py for headless GPU detection.

Replaces the runtime GPU capability detection with a static architecture
list from $TORCH_CUDA_ARCH_LIST so SageAttention compiles in Docker
build environments where no GPU is available.

Usage: cd /tmp/sageattention && python3 /tmp/patch_sageattention.py
"""
import os

arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")
arch_set = "{" + ", ".join(f'"{a}"' for a in arch_list.split(";")) + "}"

with open("setup.py", "r") as f:
    content = f.read()

old = (
    "compute_capabilities = set()\n"
    "device_count = torch.cuda.device_count()\n"
    "for i in range(device_count):\n"
    '    major, minor = torch.cuda.get_device_capability(i)\n'
    "    if major < 8:\n"
    '        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")\n'
    "        continue\n"
    '    compute_capabilities.add(f"{major}.{minor}")'
)

new = (
    f"compute_capabilities = {arch_set}\n"
    f'print(f"Manually set compute capabilities: {{compute_capabilities}}")'
)

content = content.replace(old, new)

with open("setup.py", "w") as f:
    f.write(content)

print(f"SageAttention setup.py patched for arch: {arch_set}")
