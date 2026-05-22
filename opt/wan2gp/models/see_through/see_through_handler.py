"""See-Through family handler — anime layer decomposition.

Two-stage pipeline:
1. LayerDiff: UNet denoising + TransparentVAE decode for body part extraction
2. Marigold: UNet depth estimation across all layers
3. Post-processing: sort by depth median

8 nn.Modules in mmgp pipe:
- ld_unet, ld_vae, ld_trans_vae, ld_text_encoder, ld_text_encoder_2 (LayerDiff)
- mg_unet, mg_vae, mg_text_encoder (Marigold)
"""
import gc
import io
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from models.base_handler import BaseFamilyHandler, HandlerHooks, _make_handler_cls


class _SeeThroughHooks(HandlerHooks):
    needs_bf16_autocast = True
    needs_device_patch = True

    def on_loaded(self, pipeline, pipe, base_model_type):
        # Pre-import multitalk_utils so handler's later import succeeds
        try:
            import importlib
            importlib.import_module("models.wan.multitalk.multitalk_utils")
        except (ImportError, ModuleNotFoundError):
            pass

        # Trellis uses BiRefNet (rembg wrapper) for background removal.
        # The wrapper is not an nn.Module and not in the pipe dict, so mmgp
        # doesn't manage it. Move inner model to GPU.
        rembg_wrapper = getattr(pipeline, "rembg", None)
        if rembg_wrapper is not None and torch.cuda.is_available():
            inner = getattr(rembg_wrapper, "model", None)
            if inner is not None:
                try:
                    inner.to("cuda")
                except Exception:
                    pass


HANDLER_META = {
    "input_type": "image",
    "output_type": "image",
    "hooks": _SeeThroughHooks(),
}

logger = logging.getLogger(__name__)



@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["see-through"]
    FAMILY = "see_through"
    FAMILY_INFOS = {"see_through": (403, "See-Through")}
    MODEL_DEF = {"image_outputs": True, "audio_only": False}
    DEFAULTS = {"resolution": 1280, "steps": 30}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()
        ld_path = Path(registry.get_path("image", "see-through-layerdiff"))
        mg_path = Path(registry.get_path("image", "see-through-marigold"))
        sched_path = Path(registry.get_path("image", "see-through-scheduler"))

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        seethrough_common = str(Path(cfg.project_root) / "vendor" / "seethrough" / "common")
        if seethrough_common not in sys.path:
            sys.path.insert(0, seethrough_common)

        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
        from modules.layerdiffuse.vae import TransparentVAE
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        from diffusers import DPMSolverMultistepScheduler
        from modules.marigold.marigold_depth_pipeline import MarigoldDepthPipeline

        trans_vae = TransparentVAE.from_pretrained(str(ld_path), subfolder="trans_vae")
        ld_unet = UNetFrameConditionModel.from_pretrained(str(ld_path), subfolder="unet")
        ld_pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            str(ld_path), trans_vae=trans_vae, unet=ld_unet, scheduler=None,
        )
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            str(sched_path), subfolder="scheduler",
            final_sigmas_type="zero", euler_at_final=True,
        )

        ld_vae = ld_pipeline.vae
        ld_text_encoder = ld_pipeline.text_encoder
        ld_text_encoder_2 = ld_pipeline.text_encoder_2
        ld_tokenizer = ld_pipeline.tokenizer
        ld_tokenizer_2 = ld_pipeline.tokenizer_2

        for m in [ld_vae, ld_unet, trans_vae, ld_text_encoder, ld_text_encoder_2]:
            m.to(dtype=torch.bfloat16)
            m.eval()

        mg_unet = UNetFrameConditionModel.from_pretrained(str(mg_path), subfolder="unet")
        mg_pipeline = MarigoldDepthPipeline.from_pretrained(str(mg_path), unet=mg_unet)
        mg_vae = mg_pipeline.vae
        mg_text_encoder = mg_pipeline.text_encoder
        mg_tokenizer = mg_pipeline.tokenizer
        mg_scheduler = mg_pipeline.scheduler

        for m in [mg_vae, mg_unet, mg_text_encoder]:
            m.to(dtype=torch.bfloat16)
            m.eval()

        pipe = {
            "ld_unet": ld_unet, "ld_vae": ld_vae, "ld_trans_vae": trans_vae,
            "ld_text_encoder": ld_text_encoder, "ld_text_encoder_2": ld_text_encoder_2,
            "mg_unet": mg_unet, "mg_vae": mg_vae, "mg_text_encoder": mg_text_encoder,
        }

        pipeline = _Pipeline(
            ld_unet=ld_unet, ld_vae=ld_vae, ld_trans_vae=trans_vae,
            ld_text_encoder=ld_text_encoder, ld_text_encoder_2=ld_text_encoder_2,
            ld_tokenizer=ld_tokenizer, ld_tokenizer_2=ld_tokenizer_2,
            ld_scheduler=scheduler,
            mg_unet=mg_unet, mg_vae=mg_vae, mg_text_encoder=mg_text_encoder,
            mg_tokenizer=mg_tokenizer, mg_scheduler=mg_scheduler,
        )

        co_tenants = {
            "ld_unet": ["ld_vae", "ld_trans_vae", "ld_text_encoder", "ld_text_encoder_2"],
            "mg_unet": ["mg_vae", "mg_text_encoder"],
        }

        return pipeline, {"pipe": pipe, "coTenantsMap": co_tenants}


class _Pipeline:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._empty_text_embed = None
        self._cached_prompt_embeds = {}

    def generate(self, *, image=None, resolution=1280, steps=30, seed=-1, **kwargs):
        import base64

        img_data = image
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data)).convert("RGBA")

        with torch.no_grad():
            layer_images = self._stage_layerdiff(img, resolution=resolution, steps=steps)
            depth_maps = self._stage_marigold(img, layer_images, resolution=768)

        part_dicts = self._stage_post(img, layer_images, depth_maps)
        layers = [{"name": p["tag"]} for p in part_dicts]

        return {
            "status": "success",
            "data": base64.b64encode(torch.tensor(0).numpy().tobytes()).decode(),
            "media_type": "application/json",
            "layers": layers,
        }

    def _stage_layerdiff(self, fullpage_rgba, resolution=1280, steps=30):
        from utils.cv import center_square_pad_resize
        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import sample_dpmpp_2m
        import utils.torch_utils

        input_arr = np.array(fullpage_rgba)
        fullpage, _, _ = center_square_pad_resize(input_arr, resolution, return_pad_info=True)
        c_concat = self._encode_condition_latent(fullpage)
        tag_version = self.ld_unet.get_tag_version()

        if tag_version == "v2":
            tags = ['hair','headwear','face','eyes','eyewear','ears','earwear',
                    'nose','mouth','neck','neckwear','topwear','handwear',
                    'bottomwear','legwear','footwear','tail','wings','objects']
            images = self._ld_denoise(tags=tags, c_concat=c_concat, fullpage=fullpage,
                                       steps=steps, resolution=resolution, group_index=None)
        elif tag_version == "v3":
            body_tags = ['front hair','back hair','head','neck','neckwear','topwear',
                         'handwear','bottomwear','legwear','footwear','tail','wings','objects']
            head_tags = ['headwear','face','irides','eyebrow','eyewhite','eyelash',
                         'eyewear','ears','earwear','nose','mouth']
            body_images = self._ld_denoise(tags=body_tags, c_concat=c_concat, fullpage=fullpage,
                                            steps=steps, resolution=resolution, group_index=0)
            head_img_arr = body_images[2]
            head_mask = (np.array(head_img_arr)[..., -1] > 15).astype(np.uint8)
            if np.any(head_mask):
                import cv2
                hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero(head_mask))
                input_head = input_arr[hy0:hy0+hh, hx0:hx0+hw]
            else:
                input_head = input_arr
            head_fullpage, _, _ = center_square_pad_resize(input_head, resolution, return_pad_info=True)
            head_c_concat = self._encode_condition_latent(head_fullpage)
            head_images = self._ld_denoise(tags=head_tags, c_concat=head_c_concat, fullpage=head_fullpage,
                                            steps=steps, resolution=resolution, group_index=1)
            images = body_images[:3] + head_images
        else:
            raise ValueError(f"Unknown tag_version: {tag_version}")
        return images

    def _encode_condition_latent(self, fullpage):
        from utils.torch_utils import img2tensor
        from modules.layerdiffuse.vae import vae_encode
        page_alpha = img2tensor(fullpage[..., -1] / 255.0,
                                device=self.ld_vae.device, dtype=self.ld_vae.dtype)[0][..., None]
        rgb_part = fullpage[..., :3]
        c_concat_np = np.concatenate([np.full_like(rgb_part[..., :1], 255), rgb_part], axis=2)
        c_concat_t = img2tensor(c_concat_np, normalize=True)
        c_concat = vae_encode(self.ld_vae, self.ld_trans_vae.encoder, c_concat_t, use_offset=False)
        c_concat = c_concat.to(device=self.ld_unet.device, dtype=self.ld_unet.dtype)
        self._page_alpha = page_alpha
        return c_concat

    def _encode_prompt(self, prompt):
        device = self.ld_vae.device
        embeds_list = []
        pooled = None
        for tokenizer, text_encoder in [(self.ld_tokenizer, self.ld_text_encoder),
                                          (self.ld_tokenizer_2, self.ld_text_encoder_2)]:
            ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids
            out = text_encoder(ids.to(device), output_hidden_states=True, return_dict=False)
            pooled = out[0]
            embeds_list.append(out[-1][-2])
        prompt_embeds = torch.concat(embeds_list, dim=-1).to(dtype=self.ld_unet.dtype, device=device)
        return prompt_embeds, pooled.view(prompt_embeds.shape[0], -1)

    def _ld_denoise(self, tags, c_concat, fullpage, steps, resolution, group_index):
        from utils.torch_utils import img2tensor
        device = self.ld_unet.device
        dtype = self.ld_unet.dtype
        num_frames = len(tags)
        lh, lw = c_concat.shape[-2:]

        prompt_embeds_list, pooled_list = [], []
        for tag in tags:
            if tag not in self._cached_prompt_embeds:
                pe, pp = self._encode_prompt(tag)
                self._cached_prompt_embeds[tag] = [pe.cpu(), pp.cpu()]
            pe, pp = self._cached_prompt_embeds[tag]
            prompt_embeds_list.append(pe.to(device))
            pooled_list.append(pp.to(device))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        pooled_embeds = torch.cat(pooled_list, dim=0)
        add_time_ids = torch.tensor([[lh*8, lw*8, 0, 0, lh*8, lw*8]], dtype=dtype).expand(num_frames, -1).to(device)

        if c_concat.ndim == 4:
            c_concat = c_concat[:, None].expand(-1, num_frames, -1, -1, -1)

        noise = torch.randn((1, 1, 4, lh, lw), device=device, dtype=dtype).expand(-1, num_frames, -1, -1, -1)
        self.ld_scheduler.set_timesteps(steps, device=device)
        latents = noise * self.ld_scheduler.sigmas[0]

        for t in self.ld_scheduler.timesteps:
            latent_model_input = self.ld_scheduler.scale_model_input(latents, t)
            unet_input = torch.cat([latent_model_input, c_concat.expand_as(latent_model_input[:, :, :4])[:, :, :4]], dim=2)
            noise_pred = self.ld_unet(unet_input, t, encoder_hidden_states=prompt_embeds,
                                       added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
                                       return_dict=False, group_index=group_index)[0]
            latents = self.ld_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents[0].to(dtype=self.ld_trans_vae.dtype, device=self.ld_trans_vae.device)
        latents = latents / self.ld_vae.config.scaling_factor
        page_alpha = getattr(self, "_page_alpha", None)
        images = []
        for latent in latents:
            result_list, _ = self.ld_trans_vae.decoder(self.ld_vae, latent[None], mask=page_alpha)
            images.extend(result_list)
        return images

    def _stage_marigold(self, fullpage_rgba, layer_images, resolution=768):
        from utils.cv import center_square_pad_resize, smart_resize
        from utils.torch_utils import img2tensor
        from modules.marigold.marigold_depth_pipeline import encode_argb_list

        fullpage_arr = np.array(fullpage_rgba)
        src_h, src_w = fullpage_arr.shape[:2]
        src_rescaled = resolution != src_h or resolution != src_w

        valid_tags = ['hair','headwear','face','eyes','eyewear','ears','earwear',
                      'nose','mouth','neck','neckwear','topwear','handwear',
                      'bottomwear','legwear','footwear','tail','wings','objects']
        img_arrays = []
        for tag in valid_tags[:len(layer_images)]:
            arr = np.array(layer_images[min(valid_tags.index(tag), len(layer_images) - 1)])
            # Ensure RGBA — some VAE outputs may be RGB
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
            elif arr.shape[-1] == 3:
                arr = np.concatenate([arr, np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)], axis=-1)
            arr[..., -1][arr[..., -1] < 15] = 0
            # Skip zero-dimension arrays that would break cv2.resize
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                arr = np.zeros((src_h, src_w, 4), dtype=np.uint8)
            img_arrays.append(arr)

        blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)
        for arr in img_arrays:
            if arr.shape[:2] != (src_h, src_w):
                continue
            blended_alpha += arr[..., -1].astype(np.float32) / 255
        fullpage_arr = fullpage_arr.copy()
        fullpage_arr[..., -1] = (np.clip(blended_alpha, 0, 1) * 255).astype(np.uint8)
        img_arrays.append(fullpage_arr)

        if src_rescaled:
            resized = []
            for img in img_arrays:
                try:
                    resized.append(smart_resize(img, (resolution, resolution)))
                except cv2.error:
                    resized.append(np.zeros((resolution, resolution, 4), dtype=np.uint8))
            img_arrays = resized

        img_tensors = []
        for arr in img_arrays:
            t = np.concatenate([arr[..., 3:], arr[..., :3]], axis=2).astype(np.float32) / 255.0
            img_tensors.append(img2tensor(t, dim_order="chw"))
        img_tensor_stack = torch.stack(img_tensors)

        rgb_latent_list = []
        for img_t in img_tensor_stack:
            latent = encode_argb_list(self.mg_vae, img_t[None, None].to(device=self.mg_vae.device, dtype=self.mg_vae.dtype),
                                       pad_argb=True, dtype=self.mg_vae.dtype)
            rgb_latent_list.append(latent)
        rgb_latent = torch.cat(rgb_latent_list, dim=1)

        cond_full_page = img_tensor_stack[-1][None, None]
        cond_latent_full = encode_argb_list(self.mg_vae, cond_full_page, pad_argb=True, dtype=self.mg_vae.dtype)
        ncls = len(img_arrays)
        cond_latent = torch.cat([cond_latent_full.expand(-1, ncls, -1, -1, -1), rgb_latent], dim=2)[0]

        depth_tensor = self._mg_infer(cond_latent)
        depth_pred = depth_tensor.to(device="cpu", dtype=torch.float32).numpy()
        if src_rescaled:
            depth_pred = np.array([smart_resize(d, (src_h, src_w)) for d in depth_pred])
        return [d for d in depth_pred[:-1]]

    def _mg_infer(self, cond_latent, denoising_steps=4):
        device = self.mg_unet.device
        b, c, h, w = cond_latent.shape
        self.mg_scheduler.set_timesteps(denoising_steps, device=device)

        if self._empty_text_embed is None:
            text_inputs = self.mg_tokenizer("", padding="do_not_pad", max_length=self.mg_tokenizer.model_max_length,
                                             truncation=True, return_tensors="pt")
            self._empty_text_embed = self.mg_text_encoder(text_inputs.input_ids.to(device))[0].to(self.mg_unet.dtype)

        batch_empty = self._empty_text_embed.repeat(b, 1, 1).to(device)
        target_latent = torch.randn(b, 4, h, w, device=device, dtype=self.mg_unet.dtype)

        for t in self.mg_scheduler.timesteps:
            noise_pred = self.mg_unet(torch.cat([cond_latent, target_latent], dim=1)[None],
                                       t, encoder_hidden_states=batch_empty).sample[0]
            target_latent = self.mg_scheduler.step(noise_pred, t, target_latent).prev_sample

        depth_latent = target_latent.to(device=self.mg_vae.device, dtype=self.mg_vae.dtype) / 0.18215
        z = self.mg_vae.post_quant_conv(depth_latent)
        # Decode one layer at a time to avoid OOM (20 layers * 4CH * 768*768 
        # creates 4+ GiB intermediates in VAE decoder upsampling)
        depths = []
        for i in range(b):
            zi = self.mg_vae.decoder(z[i:i+1])
            depths.append(zi.mean(dim=1, keepdim=False).clip(-1.0, 1.0))
        stacked = torch.stack(depths, dim=0)
        return (stacked + 1.0) / 2.0

    def _stage_post(self, fullpage, layer_images, depth_maps):
        parts = []
        for i, (img, depth) in enumerate(zip(layer_images, depth_maps)):
            arr = np.array(img) if isinstance(img, Image.Image) else img
            mask = arr[..., -1] > 10
            if not np.any(mask):
                continue
            parts.append({"tag": f"layer_{i}", "img": arr, "depth": depth,
                          "depth_median": float(np.median(depth[mask]))})
        parts.sort(key=lambda x: x["depth_median"], reverse=True)
        return parts
