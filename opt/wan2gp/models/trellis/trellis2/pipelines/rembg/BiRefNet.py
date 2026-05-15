# File: trellis2/pipelines/rembg/BiRefNet.py
# trellis2/pipelines/rembg/BiRefNet.py
from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import os # <--- Added

class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        target_path = model_name
        # Resolve relative paths against TRELLIS_PIPELINE_ROOT env var
        if not os.path.isabs(target_path):
            root = os.environ.get("TRELLIS_PIPELINE_ROOT", "")
            if root:
                target_path = os.path.join(root, target_path)
        if not os.path.isdir(target_path):
            target_path = os.path.join(os.getcwd(), "MODELS", "RMBG-2.0")
        if not os.path.isdir(target_path):
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "MODELS", "RMBG-2.0"))
            target_path = script_path

        if not os.path.isdir(target_path) or not os.path.exists(os.path.join(target_path, "config.json")):
            raise FileNotFoundError(
                f"RMBG model not found. Tried: {model_name}, MODELS/RMBG-2.0, {script_path}\n"
                f"Pre-download the model before starting inference."
            )

        self.model = AutoModelForImageSegmentation.from_pretrained(
            target_path, 
            trust_remote_code=True,
            local_files_only=True,
        )
        
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image