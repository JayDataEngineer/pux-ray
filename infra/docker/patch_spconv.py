"""Patch spconv setup.py to register tv::Tensor pybind11 type in spconv.core_cc.

spconv's GemmTunerSimple methods use tv::Tensor as default argument values.
Without registering tv::Tensor in the same pybind11 module (spconv.core_cc),
module import fails with "type not registered yet".

TensorViewBind from cumm generates the full tv::Tensor pybind11 class binding.
We add it to the PCCM module list so it's compiled into spconv.core_cc.
"""
import re

with open('setup.py') as f:
    c = f.read()

c = c.replace(
    'from cumm.common import CompileInfo',
    'from cumm.common import CompileInfo\nfrom cumm.tensorview_bind import TensorViewBind'
)

c = c.replace(
    'cus = [gemmtuner, convtuner,',
    'cus = [TensorViewBind(), gemmtuner, convtuner,'
)

with open('setup.py', 'w') as f:
    f.write(c)

print('Patched: added TensorViewBind to spconv.core_cc module list')
