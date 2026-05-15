# File: trellis2/pipelines/__init__.py
# trellis2/pipelines/__init__.py
import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
}

__submodules = ['samplers', 'rembg']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):
    """
    Load a pipeline from a local model directory.

    Models are pre-downloaded by the deployment system.
    Path must contain pipeline.json.
    """
    import os
    import json

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Model path not found: {path}\n"
            f"Set TRELLIS_MODEL_ID to a local directory containing pipeline.json"
        )

    config_file = os.path.join(path, "pipeline.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"pipeline.json not found at {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)

