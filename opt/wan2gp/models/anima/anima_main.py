"""Minimal Anima model implementation - avoid recursion issues."""
# Minimal imports to avoid circular dependencies
import torch

# Minimal model factory
class model_factory:
    def __init__(self, **kwargs):
        print("[Anima] Using minimal model factory - NO imports")
        self.dtype = torch.bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create dummy components to satisfy handler requirements
        self.transformer = torch.nn.Module()
        self.text_encoder = torch.nn.Module()
        self.vae = torch.nn.Module()
        self.tokenizer = None
        self.scheduler = None

    def generate(self, prompt="a cat", steps=2, seed=42, **kwargs):
        # Return dummy tensor for now - just to avoid recursion
        print(f"[Anima] Generating (minimal): {prompt}")
        torch.manual_seed(seed)
        # Return dummy RGB image [1, 3, 512, 512]
        return torch.randn(1, 3, 512, 512, device=self.device)

    @property
    def _interrupt(self):
        return False

    @_interrupt.setter
    def _interrupt(self, value):
        pass

