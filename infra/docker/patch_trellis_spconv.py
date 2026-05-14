"""Patch TRELLIS SparseTensor for spconv 2.x compatibility.

spconv 2.x removed the features setter (raises ValueError) and
torch.Tensor.reshape(0, -1) is ambiguous on empty tensors.

Fixes:
1. feats setter: use _features directly instead of blocked property
2. replace(): handle empty features in reshape
"""
import sys
from pathlib import Path

trellis_basic = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vendor/trellis2/modules/sparse/basic.py")
if not trellis_basic.exists():
    # Try relative to script location
    trellis_basic = Path("/home/user/Documents/programs/ray/vendor/trellis2/modules/sparse/basic.py")

c = trellis_basic.read_text()

# Fix 1: feats setter — bypass spconv's blocked features property
c = c.replace(
    "            self.data.features = value\n        else:\n            self.data['feats'] = value",
    "            self.data._features = value\n        else:\n            self.data['feats'] = value",
)

# Fix 2: replace() — handle empty features (reshape(0, -1) is ambiguous on empty tensors)
c = c.replace(
    "                (self.data.features if self.data.features.numel() == 0 else self.data.features.reshape(self.data.features.shape[0], -1)),",
    "                self.data.features.reshape(self.data.features.shape[0], -1) if self.data.features.numel() > 0 else (torch.zeros(0, 1, device=self.data.features.device) if self.data.features.ndim == 1 else self.data.features),",
)

trellis_basic.write_text(c)
print(f"Patched {trellis_basic}: spconv 2.x compatibility (feats setter + replace reshape)")
