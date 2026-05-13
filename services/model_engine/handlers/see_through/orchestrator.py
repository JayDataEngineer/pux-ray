"""See-Through orchestrator — explicit forward() calls on decomposed nn.Modules.
 
Re-implements the three-stage pipeline without pipeline wrapper classes:
1. stage_layerdiff() — UNet denoising + TransparentVAE decode for body part extraction
2. stage_marigold()  — 3D UNet depth estimation across all layers
3. stage_post()      — numpy/CV2 post-processing (same as vendor further_extr)
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SeeThroughOrchestrator:
    """See-Through inference via direct forward() calls on decomposed modules."""

    def __init__(self, modules):
        self.m = modules
        self._empty_text_embed = None

    def generate(
        self,
        *,
        image: Any = None,
        resolution: int = 1280,
        steps: int = 30,
        seed: int = -1,
    ) -> dict:
        import base64

        img_data = image
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data)).convert("RGBA")

        with torch.no_grad():
            layer_images = self.stage_layerdiff(img, resolution=resolution, steps=steps)
            depth_maps = self.stage_marigold(img, layer_images, resolution=768)

        part_dicts = self.stage_post(img, layer_images, depth_maps)

        layers = [{"name": p["tag"]} for p in part_dicts]

        return {
            "status": "success",
            "data": base64.b64encode(torch.tensor(0).numpy().tobytes()).decode(),
            "media_type": "application/json",
            "layers": layers,
        }

    def stage_layerdiff(self, fullpage_rgba: Image.Image, resolution: int = 1280, steps: int = 30) -> list[Image.Image]:
        import numpy as np
        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import sample_dpmpp_2m
        from utils.cv import center_square_pad_resize

        input_arr = np.array(fullpage_rgba)
        fullpage, _, _ = center_square_pad_resize(input_arr, resolution, return_pad_info=True)

        c_concat = self._encode_condition_latent(fullpage)

        tag_version = self.m.ld_unet.get_tag_version()

        if tag_version == "v2":
            tags = [
                'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear',
                'nose', 'mouth', 'neck', 'neckwear', 'topwear', 'handwear',
                'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects',
            ]
            images = self._ld_denoise(
                tags=tags, c_concat=c_concat, fullpage=fullpage,
                steps=steps, resolution=resolution, group_index=None,
            )

        elif tag_version == "v3":
            body_tags = [
                'front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear',
                'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects',
            ]
            head_tags = [
                'headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash',
                'eyewear', 'ears', 'earwear', 'nose', 'mouth',
            ]

            body_images = self._ld_denoise(
                tags=body_tags, c_concat=c_concat, fullpage=fullpage,
                steps=steps, resolution=resolution, group_index=0,
            )

            head_img_arr = body_images[2]
            head_mask = (np.array(head_img_arr)[..., -1] > 15).astype(np.uint8)
            if np.any(head_mask):
                import cv2
                hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero(head_mask))
                scale = 1.0
                hx, hy = int(hx0 * scale), int(hy0 * scale)
                input_head = input_arr[hy:hy + int(hh * scale), hx:hx + int(hw * scale)]
            else:
                input_head = input_arr

            head_fullpage, _, _ = center_square_pad_resize(input_head, resolution, return_pad_info=True)
            head_c_concat = self._encode_condition_latent(head_fullpage)

            head_images = self._ld_denoise(
                tags=head_tags, c_concat=head_c_concat, fullpage=head_fullpage,
                steps=steps, resolution=resolution, group_index=1,
            )

            canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
            images = list(body_images[:2]) + list(body_images[3:])
            images = body_images[:3] + head_images
        else:
            raise ValueError(f"Unknown tag_version: {tag_version}")

        return images

    def _encode_condition_latent(self, fullpage: np.ndarray) -> torch.Tensor:
        from utils.torch_utils import img2tensor
        from modules.layerdiffuse.vae import vae_encode

        page_alpha = img2tensor(
            fullpage[..., -1] / 255.0,
            device=self.m.ld_vae.device, dtype=self.m.ld_vae.dtype,
        )[0][..., None]

        rgb_part = fullpage[..., :3]
        c_concat_np = np.concatenate([np.full_like(rgb_part[..., :1], fill_value=255), rgb_part], axis=2)
        c_concat_t = img2tensor(c_concat_np, normalize=True)
        c_concat = vae_encode(
            self.m.ld_vae, self.m.ld_trans_vae.encoder, c_concat_t, use_offset=False,
        ).to(device=self.m.ld_unet.device, dtype=self.m.ld_unet.dtype)

        self._page_alpha = page_alpha
        return c_concat

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.m.ld_vae.device
        tokenizers = [self.m.ld_tokenizer, self.m.ld_tokenizer_2]
        text_encoders = [self.m.ld_text_encoder, self.m.ld_text_encoder_2]

        pooled = None
        embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True,
                return_dict=False,
            )
            pooled = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(embeds_list, dim=-1).to(dtype=self.m.ld_unet.dtype, device=device)
        pooled = pooled.view(bs_embed, -1)
        return prompt_embeds, pooled

    def _ld_denoise(
        self, tags: list[str], c_concat: torch.Tensor,
        fullpage: np.ndarray, steps: int, resolution: int,
        group_index: int | None,
    ) -> list[Image.Image]:
        from utils.torch_utils import img2tensor

        device = self.m.ld_unet.device
        dtype = self.m.ld_unet.dtype
        num_frames = len(tags)
        lh, lw = c_concat.shape[-2:]
        height, width = lh * 8, lw * 8

        prompt_embeds_list = []
        pooled_list = []
        for tag in tags:
            if tag not in self.m._cached_prompt_embeds:
                pe, pp = self._encode_prompt(tag)
                self.m._cached_prompt_embeds[tag] = [pe.cpu(), pp.cpu()]
            else:
                pe, pp = self.m._cached_prompt_embeds[tag]
            prompt_embeds_list.append(pe.to(device))
            pooled_list.append(pp.to(device))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        pooled_embeds = torch.cat(pooled_list, dim=0)

        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype)
        add_time_ids = add_time_ids.expand(prompt_embeds.shape[0], -1).to(device)

        if c_concat.ndim == 4:
            c_concat = c_concat[:, None].expand(-1, num_frames, -1, -1, -1)

        initial_latent_shape = (1, num_frames, 4, lh, lw)
        noise = torch.randn((1, 1, 4, lh, lw), device=device, dtype=dtype)
        noise = noise.expand(-1, num_frames, -1, -1, -1)

        self.m.ld_scheduler.set_timesteps(steps, device=device)
        sigmas = self.m.ld_scheduler.sigmas
        latents = noise * sigmas[0]

        guidance_scale = 1.0
        do_cfg = guidance_scale > 1.0

        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

        for i, t in enumerate(self.m.ld_scheduler.timesteps):
            latent_model_input = latents
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)

            latent_model_input = self.m.ld_scheduler.scale_model_input(latent_model_input, t)

            unet_input = torch.cat([latent_model_input, c_concat.expand_as(
                latent_model_input[:, :, :4]
            )[:, :, :4]], dim=2)

            noise_pred = self.m.ld_unet(
                unet_input, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                group_index=group_index,
            )[0]

            latents = self.m.ld_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents[0]
        latents = latents.to(dtype=self.m.ld_trans_vae.dtype, device=self.m.ld_trans_vae.device)
        latents = latents / self.m.ld_vae.config.scaling_factor

        page_alpha = getattr(self, "_page_alpha", None)
        images = []
        for latent in latents:
            latent = latent[None]
            result_list, _ = self.m.ld_trans_vae.decoder(self.m.ld_vae, latent, mask=page_alpha)
            images.extend(result_list)

        return images

    def stage_marigold(
        self, fullpage_rgba: Image.Image, layer_images: list[Image.Image],
        resolution: int = 768,
    ) -> list[np.ndarray]:
        import numpy as np
        from utils.cv import center_square_pad_resize, smart_resize, img_alpha_blending
        from utils.torch_utils import img2tensor
        from utils.torchcv import pad_rgb_torch

        fullpage_arr = np.array(fullpage_rgba)
        src_h, src_w = fullpage_arr.shape[:2]
        src_rescaled = resolution != src_h or resolution != src_w

        valid_tags = [
            'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear',
            'nose', 'mouth', 'neck', 'neckwear', 'topwear', 'handwear',
            'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects',
        ]

        img_arrays = []
        for tag in valid_tags[:len(layer_images)]:
            arr = np.array(layer_images[min(valid_tags.index(tag), len(layer_images) - 1)])
            arr[..., -1][arr[..., -1] < 15] = 0
            img_arrays.append(arr)

        blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)
        for arr in img_arrays:
            blended_alpha += arr[..., -1].astype(np.float32) / 255
        blended_alpha = np.clip(blended_alpha, 0, 1) * 255
        blended_alpha = blended_alpha.astype(np.uint8)
        fullpage_arr = fullpage_arr.copy()
        fullpage_arr[..., -1] = blended_alpha
        img_arrays.append(fullpage_arr)

        ncls = len(img_arrays)

        if src_rescaled:
            img_arrays = [smart_resize(img, (resolution, resolution)) for img in img_arrays]

        img_tensors = []
        for arr in img_arrays:
            t = np.concatenate([arr[..., 3:], arr[..., :3]], axis=2).astype(np.float32) / 255.0
            t = img2tensor(t, dim_order="chw")
            img_tensors.append(t)
        img_tensor_stack = torch.stack(img_tensors)

        cond_full_page = img_tensor_stack[-1][None]

        from modules.marigold.marigold_depth_pipeline import encode_argb_list
        rgb_latent_list = []
        for img_t in img_tensor_stack:
            latent = encode_argb_list(
                self.m.mg_vae,
                img_t[None, None].to(device=self.m.mg_vae.device, dtype=self.m.mg_vae.dtype),
                pad_argb=True, dtype=self.m.mg_vae.dtype,
            )
            rgb_latent_list.append(latent)
        rgb_latent = torch.cat(rgb_latent_list, dim=1)

        cond_latent_full = encode_argb_list(
            self.m.mg_vae,
            cond_full_page[None],
            pad_argb=True, dtype=self.m.mg_vae.dtype,
        )
        cond_latent = torch.cat([
            cond_latent_full.expand(-1, ncls, -1, -1, -1),
            rgb_latent,
        ], dim=2)
        cond_latent = cond_latent[0]

        depth_tensor = self._mg_single_infer(cond_latent)

        depth_pred = depth_tensor.to(device="cpu", dtype=torch.float32).numpy()
        if src_rescaled:
            depth_pred = np.array([smart_resize(d, (src_h, src_w)) for d in depth_pred])

        return [d for d in depth_pred[:-1]]

    def _mg_single_infer(self, cond_latent: torch.Tensor, denoising_steps: int = 4) -> torch.Tensor:
        device = self.m.mg_unet.device
        b, c, h, w = cond_latent.shape

        self.m.mg_scheduler.set_timesteps(denoising_steps, device=device)
        timesteps = self.m.mg_scheduler.timesteps

        if self._empty_text_embed is None:
            text_inputs = self.m.mg_tokenizer(
                "", padding="do_not_pad",
                max_length=self.m.mg_tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            self._empty_text_embed = self.m.mg_text_encoder(text_input_ids)[0].to(self.m.mg_unet.dtype)

        batch_empty = self._empty_text_embed.repeat(b, 1, 1).to(device)
        target_latent = torch.randn(b, 4, h, w, device=device, dtype=self.m.mg_unet.dtype)

        for t in timesteps:
            unet_input = torch.cat([cond_latent, target_latent], dim=1)
            unet_input = unet_input[None]

            noise_pred = self.m.mg_unet(
                unet_input, t, encoder_hidden_states=batch_empty,
            ).sample

            noise_pred = noise_pred[0]
            target_latent = self.m.mg_scheduler.step(
                noise_pred, t, target_latent,
            ).prev_sample

        depth_latent = target_latent.to(device=self.m.mg_vae.device, dtype=self.m.mg_vae.dtype)
        depth_latent = depth_latent / 0.18215
        z = self.m.mg_vae.post_quant_conv(depth_latent)
        stacked = self.m.mg_vae.decoder(z)
        depth = stacked.mean(dim=1, keepdim=False)
        depth = depth.clip(-1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return depth

    def stage_post(
        self, fullpage: Image.Image, layer_images: list[Image.Image],
        depth_maps: list[np.ndarray],
    ) -> list[dict]:
        import numpy as np
        from utils.torchcv import cluster_inpaint_part

        parts = []
        for i, (img, depth) in enumerate(zip(layer_images, depth_maps)):
            arr = np.array(img) if isinstance(img, Image.Image) else img
            mask = arr[..., -1] > 10
            if not np.any(mask):
                continue
            depth_median = float(np.median(depth[mask]))
            parts.append({
                "tag": f"layer_{i}",
                "img": arr,
                "depth": depth,
                "depth_median": depth_median,
            })

        parts.sort(key=lambda x: x["depth_median"], reverse=True)
        return parts
