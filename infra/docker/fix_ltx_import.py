"""Fix circular import in ltx_core."""
p = "/usr/local/lib/python3.10/dist-packages/ltx_core/loader/fuse_loras.py"
with open(p) as f:
    c = f.read()
c = c.replace(
    "from ltx_core.quantization.fp8_cast import calculate_weight_float8",
    "try:\n    from ltx_core.quantization.fp8_cast import calculate_weight_float8\nexcept ImportError:\n    def calculate_weight_float8(*a, **k):\n        from ltx_core.quantization.fp8_cast import calculate_weight_float8 as f\n        return f(*a, **k)"
)
with open(p, "w") as f:
    f.write(c)
print("Fixed ltx_core circular import")
