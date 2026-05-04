"""Create cumesh stub package so API starts without GPU extensions.
cumesh won't compile — this stub provides the minimal API surface for pipeline init.
"""
import os

site = "/usr/local/lib/python3.11/dist-packages"

# cumesh stub only (flex_gemm is installed from GitHub with triton compat patches)
os.makedirs(f"{site}/cumesh", exist_ok=True)
open(f"{site}/cumesh/__init__.py", "w").write("")

print("cumesh stub created")
