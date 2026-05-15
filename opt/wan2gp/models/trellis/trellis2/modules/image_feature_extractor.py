# File: trellis2/modules/image_feature_extractor.py
# trellis2/modules/image_feature_extractor.py
import os
from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image

class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        # model_name must be a local path (models pre-downloaded by deployment system)
        target = model_name
        if not os.path.isdir(target):
            target = os.path.join(os.getcwd(), "MODELS", "dinov3")
        if not os.path.isdir(target) or not os.path.exists(os.path.join(target, "config.json")):
            raise FileNotFoundError(
                f"DINOv3 model not found at: {model_name}\n"
                f"Pre-download the model before starting inference."
            )

        self.model = DINOv3ViTModel.from_pretrained(target, local_files_only=True)
        self.model.eval()
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def to(self, device):
        self.model.to(device)
        self._norm_mean = self._norm_mean.to(device, non_blocking=True)
        self._norm_std = self._norm_std.to(device, non_blocking=True)

    def cuda(self):
        self.model.cuda()
        self._norm_mean = self._norm_mean.cuda()
        self._norm_std = self._norm_std.cuda()

    def cpu(self):
        self.model.cpu()
        self._norm_mean = self._norm_mean.to('cpu', non_blocking=True)
        self._norm_std = self._norm_std.to('cpu', non_blocking=True)
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self._norm_mean.device, non_blocking=True)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = (image - self._norm_mean) / self._norm_std
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        target = model_name
        # Resolve relative paths against TRELLIS_PIPELINE_ROOT env var
        if not os.path.isabs(target):
            root = os.environ.get("TRELLIS_PIPELINE_ROOT", "")
            if root:
                target = os.path.join(root, target)
        if not os.path.isdir(target):
            target = os.path.join(os.getcwd(), "MODELS", "dinov3")
        if not os.path.isdir(target):
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MODELS", "dinov3"))
            target = script_path

        if not os.path.isdir(target) or not os.path.exists(os.path.join(target, "config.json")):
            raise FileNotFoundError(
                f"DINOv3 model not found. Tried: {model_name}, MODELS/dinov3, {script_path}\n"
                f"Pre-download the model before starting inference."
            )

        self.model = DINOv3ViTModel.from_pretrained(target, local_files_only=True)
        
        self.model.eval()
        self.image_size = image_size
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def to(self, device):
        self.model.to(device)
        self._norm_mean = self._norm_mean.to(device, non_blocking=True)
        self._norm_std = self._norm_std.to(device, non_blocking=True)

    def cuda(self):
        self.model.cuda()
        self._norm_mean = self._norm_mean.cuda()
        self._norm_std = self._norm_std.cuda()

    def cpu(self):
        self.model.cpu()
        self._norm_mean = self._norm_mean.to('cpu', non_blocking=True)
        self._norm_std = self._norm_std.to('cpu', non_blocking=True)

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self._norm_mean.device, non_blocking=True)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = (image - self._norm_mean) / self._norm_std
        features = self.extract_features(image)
        return features