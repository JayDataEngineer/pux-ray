"""Patches SageAttention setup.py for headless GPU detection.

Replaces the runtime GPU capability detection with a static architecture
list from $TORCH_CUDA_ARCH_LIST so SageAttention compiles in Docker
build environments where no GPU is available.
"""
import os
import urllib.request

arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.6")
arch_set = "{" + ", ".join(f'"{a}"' for a in arch_list.split(";")) + "}"

url = "https://raw.githubusercontent.com/thu-ml/SageAttention/main/setup.py"
with urllib.request.urlopen(url) as r:
    content = r.read().decode()

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

with open("/tmp/setup_patched.py", "w") as f:
    f.write(content)

print(f"SageAttention setup.py patched for arch: {arch_set}")
