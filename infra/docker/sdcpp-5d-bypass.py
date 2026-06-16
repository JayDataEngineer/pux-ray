#!/usr/bin/env python3
"""Patch gguf.cpp to bypass the 4D tensor dimension limit for Wan VACE."""
import re, sys

path = 'ggml/src/gguf.cpp'
src = open(path).read()

# Turn the 4D error check into a no-op warning:
#   ok = false; break;  →  ok = true;
src = re.sub(
    r'(if \(n_dims > GGML_MAX_DIMS\) \{\s*\n\s*GGML_LOG_ERROR[^}]*?)\s*ok = false;\s*\n\s*break;\s*\n(\s*\})',
    r'\1ok = true;\2', src,
    flags=re.DOTALL
)

count = src.count('ok = true;') - open(path).read().count('ok = false')
open(path, 'w').write(src)
print(f'Patched gguf.cpp: {count} checks bypassed')
