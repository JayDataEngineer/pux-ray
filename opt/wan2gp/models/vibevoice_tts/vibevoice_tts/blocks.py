"""Authored VibeVoice building blocks: RMSNorm, ConvNeXt blocks, VQ codec."""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f * rms
        if self.elementwise_affine:
            out = out * self.weight.float()
        return out.to(x.dtype)


class ConvLayerNorm(nn.Module):
    """LayerNorm with channel-last transpose for conv features."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.layer_norm(x.float(), self.weight.shape, self.weight.float(),
                         self.bias.float(), self.eps)
        return x.transpose(1, 2).to(x.dtype)


class ConvRMSNorm(nn.Module):
    """RMSNorm with channel-last transpose for conv features."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, 2).float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms * self.weight.float()
        return out.transpose(1, 2).to(x.dtype)


class SConv1d(nn.Module):
    """Scalable conv1d with weight norm and causal padding."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, causal=True, pad_mode='reflect'):
        super().__init__()
        self.causal = causal
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        k = self.conv.kernel_size[0]
        d = self.conv.dilation[0]
        if self.causal:
            pad = (k - 1) * d
            x = F.pad(x, (pad, 0))
        else:
            pad = (k - 1) * d // 2
            x = F.pad(x, (pad, pad), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Scalable transposed conv1d for decoder upsampling."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True,
                 causal=True):
        super().__init__()
        self.causal = causal
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                       stride=stride, bias=bias)
        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        return self.conv(x)


class FFN(nn.Module):
    """Feed-forward network: Linear → GELU → Linear."""

    def __init__(self, dim, ffn_dim, bias=False):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, dim, bias=bias)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class Convlayer(nn.Module):
    """Depthwise/group conv layer with scaling."""

    def __init__(self, dim, kernel_size=7, groups=1, causal=True,
                 pad_mode='reflect', bias=True):
        super().__init__()
        self.conv = SConv1d(dim, dim, kernel_size, groups=groups,
                           causal=causal, pad_mode=pad_mode, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Block1D(nn.Module):
    """ConvNeXt V2-style block: RMSNorm → Depthwise Conv → RMSNorm → FFN."""

    def __init__(self, dim, kernel_size=7, mixer_layer='depthwise_conv',
                 layer_scale_init_value=1e-6, ffn_expansion=4,
                 norm='RMSNorm', causal=True, pad_mode='reflect', eps=1e-5):
        super().__init__()
        if norm == 'LN':
            self.norm = ConvLayerNorm(dim, eps=eps)
            self.ffn_norm = ConvLayerNorm(dim, eps=eps)
        else:
            self.norm = ConvRMSNorm(dim, eps=eps)
            self.ffn_norm = ConvRMSNorm(dim, eps=eps)

        groups = dim if mixer_layer == 'depthwise_conv' else 1
        self.mixer = Convlayer(dim, kernel_size=kernel_size, groups=groups,
                               causal=causal, pad_mode=pad_mode, bias=True)
        self.ffn = FFN(dim, ffn_expansion * dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init_value)\
            if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        x_ffn = self.ffn_norm(x).transpose(1, 2)
        x_ffn = self.ffn(x_ffn).transpose(1, 2)
        if self.gamma is not None:
            x_ffn = x_ffn * self.gamma.view(1, -1, 1)
        x = x + x_ffn
        return x


class EncoderStage(nn.Module):
    """One encoder stage: N blocks + strided conv for downsampling."""

    def __init__(self, in_ch, out_ch, num_blocks, stride_ratio,
                 encoder_n_filters=32, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Block1D(in_ch, **kwargs))
        self.blocks = nn.Sequential(*blocks)
        self.downsample = SConv1d(in_ch, out_ch, kernel_size=stride_ratio * 2,
                                  stride=stride_ratio, causal=kwargs.get('causal', True))

    def forward(self, x):
        return self.downsample(self.blocks(x))


class DecoderStage(nn.Module):
    """One decoder stage: transposed conv + N blocks."""

    def __init__(self, in_ch, out_ch, num_blocks, stride_ratio,
                 **kwargs):
        super().__init__()
        self.upsample = SConvTranspose1d(in_ch, out_ch,
                                         kernel_size=stride_ratio * 2,
                                         stride=stride_ratio,
                                         causal=kwargs.get('causal', True))
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Block1D(out_ch, **kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.upsample(x))


class VibeVoiceAcousticTokenizer(nn.Module):
    """Full codec VAE: encoder → latent → decoder.

    Encoder: 6 conv stages with increasing channels.
    Latent: VQ-like with Gaussian sampling.
    Decoder: 6 transposed conv stages (mirror of encoder).
    """

    def __init__(self, config):
        super().__init__()
        cfg = config
        n_filters = cfg.get('encoder_n_filters', 32)
        ratios = cfg.get('encoder_ratios', [8, 5, 5, 4, 2, 2])
        depths = [int(d) for d in cfg.get('encoder_depths', '3-3-3-3-3-3-8').split('-')]
        vae_dim = cfg.get('vae_dim', 64)
        causal = cfg.get('causal', True)
        norm = cfg.get('layernorm', 'RMSNorm')
        eps = cfg.get('layernorm_eps', 1e-5)
        mixer = cfg.get('mixer_layer', 'depthwise_conv')

        kwargs = dict(causal=causal, norm=norm, pad_mode=cfg.get('pad_mode', 'constant'),
                      eps=eps, mixer_layer=mixer)

        # Encoder
        ch = n_filters
        enc_stages = []
        ch_in = 1
        for i, (r, d) in enumerate(zip(ratios, depths)):
            ch_out = ch * (2 ** i) if i < 4 else ch * 16
            enc_stages.append(EncoderStage(ch_in, ch_out, d, r,
                              encoder_n_filters=n_filters, **kwargs))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_stages)
        enc_dim = ch_in

        # Project to VAE latent
        self.encoder_proj = nn.Conv1d(enc_dim, vae_dim * 2, 1)
        self.std_dist_type = cfg.get('std_dist_type', 'gaussian')
        self.fix_std = cfg.get('fix_std', 0.5)

        # Decoder (mirrors encoder)
        dec_stages = []
        for i, (r, d) in enumerate(reversed(list(zip(ratios, depths)))):
            idx = len(ratios) - 1 - i
            ch_in = ch * (2 ** idx) if idx < 4 else ch * 16
            ch_out = 1 if i == len(ratios) - 1 else (ch * (2 ** (idx - 1)) if idx > 0 else ch)
            num_blk = depths[idx]
            dec_stages.append(DecoderStage(ch_in, ch_out, num_blk, r, **kwargs))
        self.decoder = nn.Sequential(*dec_stages)

        self.sample_rate = cfg.get('sample_rate', 24000)

    def encode(self, x):
        """Encode audio waveform to latent codes."""
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        enc = self.encoder(x)
        enc = self.encoder_proj(enc)
        mean, logvar = enc.chunk(2, dim=1)
        if self.std_dist_type == 'gaussian':
            std = logvar.mul(0.5).exp() if self.fix_std == 0 else torch.ones_like(logvar) * self.fix_std
            noise = torch.randn_like(mean)
            z = mean + noise * std
        else:
            z = mean
        return z

    def decode(self, z):
        """Decode latent codes to audio waveform."""
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class VibeVoiceSemanticTokenizer(nn.Module):
    """Encoder-only semantic tokenizer."""

    def __init__(self, config):
        super().__init__()
        cfg = config
        n_filters = cfg.get('encoder_n_filters', 32)
        ratios = cfg.get('encoder_ratios', [8, 5, 5, 4, 2, 2])
        depths = [int(d) for d in cfg.get('encoder_depths', '3-3-3-3-3-3-8').split('-')]
        vae_dim = cfg.get('vae_dim', 64)
        causal = cfg.get('causal', True)
        norm = cfg.get('layernorm', 'RMSNorm')
        eps = cfg.get('layernorm_eps', 1e-5)
        mixer = cfg.get('mixer_layer', 'depthwise_conv')

        kwargs = dict(causal=causal, norm=norm, pad_mode=cfg.get('pad_mode', 'constant'),
                      eps=eps, mixer_layer=mixer)

        ch = n_filters
        enc_stages = []
        ch_in = 1
        for i, (r, d) in enumerate(zip(ratios, depths)):
            ch_out = ch * (2 ** i) if i < 4 else ch * 16
            enc_stages.append(EncoderStage(ch_in, ch_out, d, r,
                              encoder_n_filters=n_filters, **kwargs))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_stages)
        self.encoder_proj = nn.Conv1d(ch_in, vae_dim, 1)
        self.std_dist_type = cfg.get('std_dist_type', 'none')

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        enc = self.encoder(x)
        return self.encoder_proj(enc)


class SpeechConnector(nn.Module):
    """Linear → RMSNorm → Linear projection for speech features."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x
