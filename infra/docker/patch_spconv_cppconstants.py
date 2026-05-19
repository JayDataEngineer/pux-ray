"""Patch spconv cppconstants.py to import cumm.core_cc before spconv.core_cc.

spconv's core_cc module has GemmTunerSimple methods with tv::Tensor default
arguments, but tv::Tensor isn't registered in spconv's pybind11 module.
cumm.core_cc DOES register it. Both modules share pybind11 internals, so
pre-importing cumm.core_cc makes the type visible to spconv.core_cc.
"""
import spconv.cppconstants as cc
with open(cc.__file__) as f:
    c = f.read()
old = "import spconv.core_cc as _ext"
new = "import cumm.core_cc\nimport spconv.core_cc as _ext"
c = c.replace(old, new)
with open(cc.__file__, 'w') as f:
    f.write(c)
print("Patched spconv.cppconstants: added import cumm.core_cc before spconv.core_cc")
