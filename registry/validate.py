"""IaC validation gate — catches config drift before it reaches the cluster.

Runs in CI (Gitea Actions) and locally (task validate). Checks:
1. YAML syntax
2. VRAM budget (model + KV cache + draft < GPU capacity)
3. Cross-references (FORGE_SERVICES, DEFAULT_MODEL, registry agree)
4. TNAP protocol completeness (services get the fields they need)
5. No hardcoded model names in deployment code that diverge from registry

Usage:
    python -m registry.validate
    # exits 0 = pass, 1 = failures found
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "model_registry.yaml"

RTX_4090_VRAM_GB = 24.0
# Leave 2GB for compute buffers, flash attention, CUDA overhead
VRAM_USABLE_GB = RTX_4090_VRAM_GB - 2.0


class ValidationFail:
    def __init__(self, check: str, message: str):
        self.check = check
        self.message = message

    def __str__(self):
        return f"[{self.check}] {self.message}"


def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def _estimate_kv_cache_gb(meta: dict) -> float:
    """Estimate KV cache size for a given model config.

    Formula: 2 * n_layers * n_kv_heads * head_dim * bytes_per_element * ctx_size
    Uses quantization type from cache_type_k/v to estimate bytes per element.
    """
    ctx_size = meta.get("ctx_size", 8192)
    cache_type_k = meta.get("cache_type_k", "f16")
    cache_type_v = meta.get("cache_type_v", "f16")

    # bytes per element for different cache types
    cache_bytes = {
        "f16": 2, "q8_0": 1, "q4_0": 0.5, "q4_1": 0.5,
        "turbo4": 0.5, "turbo3": 0.5, "turbo3_tcq": 0.5,
        "q8_1": 1, "iq4_nl": 0.5, "iq4_xs": 0.5,
    }

    k_bytes = cache_bytes.get(cache_type_k, 2)
    v_bytes = cache_bytes.get(cache_type_v, 2)

    # Estimate model architecture from size.
    # This is a rough heuristic — for exact values we'd need to load the GGUF.
    # Qwen 27B: 64 layers, 4 kv_heads, 256 head_dim
    # Qwen 3B: 36 layers, 4 kv_heads, 128 head_dim
    size_gb = meta.get("size_gb", 0)
    if size_gb >= 15:
        n_layers, n_kv_heads, head_dim = 64, 4, 256
    elif size_gb >= 5:
        n_layers, n_kv_heads, head_dim = 36, 4, 128
    else:
        n_layers, n_kv_heads, head_dim = 24, 4, 64

    per_token = n_layers * (n_kv_heads * head_dim * k_bytes +
                            n_kv_heads * head_dim * v_bytes)
    total_bytes = per_token * ctx_size
    return total_bytes / (1024 ** 3)


def check_yaml_syntax() -> list[ValidationFail]:
    """YAML parses without errors."""
    failures = []
    try:
        with open(_REGISTRY_PATH) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        failures.append(ValidationFail("yaml_syntax", str(e)))
    return failures


def _get_autoloaded_models(data: dict) -> set[tuple[str, str]]:
    """Models that would actually be loaded by services (DEFAULT_MODEL + Forge SERVICE_MAP)."""
    autoloaded = set()
    # 1. DEFAULT_MODEL from deployment files
    for deploy_dir in (_PROJECT_ROOT / "services").rglob("deployment.py"):
        src = deploy_dir.read_text()
        for m in re.finditer(r'DEFAULT_MODEL\s*=\s*"([^"]+)"', src):
            model_name = m.group(1)
            for cat, models in data.items():
                if isinstance(models, dict) and model_name in models:
                    autoloaded.add((cat, model_name))

    # 2. Forge SERVICE_MAP defaults (from services/forge.py)
    forge_path = _PROJECT_ROOT / "services" / "forge.py"
    if forge_path.exists():
        src = forge_path.read_text()
        for m in re.finditer(r'"model_name":\s*"([^"]+)"', src):
            model_name = m.group(1)
            for cat, models in data.items():
                if isinstance(models, dict) and model_name in models:
                    autoloaded.add((cat, model_name))

    return autoloaded


def check_vram_budget(data: dict) -> list[ValidationFail]:
    """Auto-loaded GPU models fit within usable VRAM (24GB - 2GB overhead)."""
    failures = []
    autoloaded = _get_autoloaded_models(data)

    for cat, name in autoloaded:
        meta = data[cat][name]
        if meta.get("device") != "gpu":
            continue

        # vram_estimate_gb should include model tensors + KV cache + compute buffers.
        # We add draft model on top since it's listed separately in the registry.
        vram_est = meta.get("vram_estimate_gb", meta.get("size_gb", 0))

        # Add draft model VRAM if speculative decoding
        draft_gb = 0
        draft_model = meta.get("spec_draft_model")
        if draft_model:
            draft_name = Path(draft_model).stem
            for dcat, dmodels in data.items():
                if not isinstance(dmodels, dict):
                    continue
                for dname, dmeta in dmodels.items():
                    if not isinstance(dmeta, dict):
                        continue
                    dpath = dmeta.get("path", "")
                    if draft_model in dpath or draft_name in dpath:
                        draft_gb = dmeta.get("size_gb", 0)
                        break
                if draft_gb > 0:
                    break

        total = vram_est + draft_gb
        if total > VRAM_USABLE_GB:
            failures.append(ValidationFail(
                "vram_budget",
                f"{cat}/{name}: estimated {total:.1f}GB > {VRAM_USABLE_GB:.0f}GB usable "
                f"(vram_estimate={vram_est:.1f}GB + draft={draft_gb:.1f}GB). "
                f"Reduce vram_estimate_gb, ctx_size, or disable DFlash."
            ))
    return failures


def check_cross_references(data: dict) -> list[ValidationFail]:
    """Forge SERVICE_MAP and DEFAULT_MODEL reference models that exist in the registry."""
    failures = []

    # 1. Check Forge SERVICE_MAP default models
    forge_path = _PROJECT_ROOT / "services" / "forge.py"
    if forge_path.exists():
        src = forge_path.read_text()
        # Extract any hardcoded model references
        for m in re.finditer(r'"model_name":\s*"([^"]+)"', src):
            model_name = m.group(1)
            found = False
            for cat, models in data.items():
                if isinstance(models, dict) and model_name in models:
                    found = True
                    break
            if not found:
                failures.append(ValidationFail(
                    "cross_reference",
                    f"Forge references model '{model_name}' "
                    f"but it doesn't exist in model_registry.yaml"
                ))

    # 2. Check deployment DEFAULT_MODEL values
    llm_deploy_path = _PROJECT_ROOT / "services" / "llm" / "deployment.py"
    if llm_deploy_path.exists():
        src = llm_deploy_path.read_text()
        for m in re.finditer(r'DEFAULT_MODEL\s*=\s*"([^"]+)"', src):
            model_name = m.group(1)
            found = "llm" in data and model_name in data.get("llm", {})
            if not found:
                failures.append(ValidationFail(
                    "cross_reference",
                    f"LLMDeployment DEFAULT_MODEL='{model_name}' "
                    f"not found in registry llm section"
                ))

    # 3. No longer need LOAD_KWARGS vs DEFAULT_MODEL mismatch check
    # The Forge reads default_model from each ForgeService class directly

    return failures


def check_tnap_fields(data: dict) -> list[ValidationFail]:
    """Services that use messages/stream can receive them through TNAP."""
    failures = []

    tnap_path = _PROJECT_ROOT / "services" / "base.py"
    if not tnap_path.exists():
        return failures

    src = tnap_path.read_text()

    # TNAP protocol removed — skip if not present
    if "class TNAPInput" not in src:
        return failures

    # If TNAPInput still exists (legacy), validate it has required fields
    tnap_section = src.split("class TNAPInput")[1].split("class ")[0]
    if "messages" not in tnap_section:
        failures.append(ValidationFail(
            "tnap_protocol",
            "TNAPInput missing 'messages' field — LLM chat won't work"
        ))
    if "stream:" not in tnap_section:
        failures.append(ValidationFail(
            "tnap_protocol",
            "TNAPInput missing 'stream' field — streaming won't work"
        ))

    return failures


def check_description_syntax(data: dict) -> list[ValidationFail]:
    """Description fields with colons are quoted (YAML safety)."""
    failures = []
    try:
        with open(_REGISTRY_PATH) as f:
            raw = f.read()
    except Exception:
        return failures

    # Re-parse to get line numbers for context
    for i, line in enumerate(raw.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("description:"):
            desc_val = stripped[len("description:"):].strip()
            # Multi-line descriptions that continue on next lines
            # Check if unquoted value contains a colon followed by space
            if desc_val and not desc_val.startswith('"') and not desc_val.startswith("'"):
                if desc_val.startswith("|") or desc_val.startswith(">"):
                    continue  # literal/folded block — safe
                if ": " in desc_val:
                    failures.append(ValidationFail(
                        "yaml_safety",
                        f"line {i}: unquoted description contains ': ' — "
                        f"YAML may misparse as mapping. Wrap in quotes."
                    ))
    return failures


def run_all() -> list[ValidationFail]:
    """Run all validation checks. Returns list of failures."""
    failures = []

    # YAML syntax (must pass before other checks)
    failures.extend(check_yaml_syntax())
    if failures:
        return failures  # Can't proceed with broken YAML

    data = _load_registry()

    failures.extend(check_vram_budget(data))
    failures.extend(check_cross_references(data))
    failures.extend(check_tnap_fields(data))
    failures.extend(check_description_syntax(data))

    return failures


def main():
    failures = run_all()
    if failures:
        print(f"FAIL — {len(failures)} validation error(s):\n")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("PASS — all validation checks OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
