"""Authored VibeVoice diffusion prediction head."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP."""

    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


class VibeVoiceDiffusionHead(nn.Module):
    """DDPM prediction head: timestep-conditioned MLP for speech denoising.

    Architecture:
        noisy_images_proj → cond_proj → N× [MLP block] → final_layer
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_size', 3584)
        latent = config.get('latent_size', 64)
        n_layers = config.get('head_layers', 4)
        ffn_ratio = config.get('head_ffn_ratio', 3.0)

        self.noisy_images_proj = nn.Linear(latent, hidden)
        self.cond_proj = nn.Linear(hidden, hidden)
        self.t_embedder = TimestepEmbedder(hidden)

        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            ])
        self.layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_ratio),
            nn.SiLU(),
            nn.Linear(int(hidden * ffn_ratio), latent),
        )

    def forward(self, latents, t, cond):
        t_emb = self.t_embedder(t)
        h = self.noisy_images_proj(latents)
        h = h + self.cond_proj(cond) + t_emb.unsqueeze(1)
        h = self.layers(h)
        return self.final_layer(h)
