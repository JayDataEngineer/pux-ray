"""Model Engine — universal PyTorch model execution with mmgp VRAM management.

Every model family gets a handler that decomposes it into nn.Module components.
mmgp manages VRAM/CPU/RAM placement. Models persist across batch tasks.
"""
