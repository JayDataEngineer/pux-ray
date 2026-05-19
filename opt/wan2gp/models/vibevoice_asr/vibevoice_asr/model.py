"""VibeVoice ASR - model matching checkpoint keys exactly."""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vibevoice_asr.vibevoice_asr.blocks import SConv1d, ConvRMSNorm, FFN, SpeechConnector

class DepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size=7, causal=True, pad_mode="constant"):
        super().__init__()
        self.conv = SConv1d(dim, dim, kernel_size, groups=dim, causal=causal, pad_mode=pad_mode)
    def forward(self, x):
        return self.conv(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, ffn_expansion=4, causal=True, pad_mode="constant", eps=1e-5):
        super().__init__()
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.mixer = nn.Sequential()
        self.mixer.add_module("conv", DepthwiseConv(dim, causal=causal, pad_mode=pad_mode))
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)
        self.ffn = FFN(dim, ffn_expansion * dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init_value)
        self.ffn_gamma = nn.Parameter(torch.ones(dim) * layer_scale_init_value)
    def forward(self, x):
        x = x + self.mixer[0](self.norm(x)) * self.gamma.view(1, -1, 1)
        x_ffn = self.ffn_norm(x).transpose(1, 2)
        x_ffn = self.ffn(x_ffn).transpose(1, 2) * self.ffn_gamma.view(1, -1, 1)
        x = x + x_ffn
        return x

class vvConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, causal=True, pad_mode="constant"):
        super().__init__()
        self.conv = SConv1d(in_ch, out_ch, kernel_size, stride=stride, causal=causal, pad_mode=pad_mode)
    def forward(self, x):
        return self.conv(x)

class TokenizerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.get("encoder_n_filters", 32)
        raw_depths = cfg.get("encoder_depths", "3-3-3-3-3-3-8")
        if isinstance(raw_depths, str):
            depths = [int(d) for d in raw_depths.split("-")]
        else:
            depths = list(raw_depths)
        vae = cfg.get("vae_dim", 64)
        causal = cfg.get("causal", True)
        pm = cfg.get("pad_mode", "constant")
        eps = cfg.get("layernorm_eps", 1e-5)
        # Kernel sizes and strides from checkpoint (7 stages)
        n_stages = len(depths)
        strides = [8, 5, 5, 4, 2, 2, 1][:n_stages]
        kernels = [7, 4, 4, 8, 10, 10, 16][:n_stages]
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        ci = 1
        for i in range(n_stages):
            co = nf * (2**i)
            self.downsample_layers.append(nn.ModuleList([vvConv1d(ci, co, kernels[i], stride=strides[i], causal=causal, pad_mode=pm)]))
            dim = nf * (2**i)
            blks = nn.ModuleList()
            for _ in range(depths[min(i, len(depths)-1)]):
                blks.append(ConvNeXtBlock(dim, causal=causal, pad_mode=pm, eps=eps))
            self.stages.append(blks)
            ci = co
        ld = nf * (2**(n_stages-1))
        self.head = nn.Sequential()
        self.head.add_module("conv", SConv1d(ld, vae, 7, causal=causal, pad_mode=pm))
    def forward(self, x):
        for dl, st in zip(self.downsample_layers, self.stages):
            x = dl[0](x)
            for b in st:
                x = b(x)
        return self.head(x)

class VibeVoiceAsrModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super().__init__()
        self._dtype, self._device = dtype, torch.device("cpu")
        acfg = config.get("acoustic_tokenizer_config", config)
        scfg = config.get("semantic_tokenizer_config", config)
        self.acoustic_tokenizer = TokenizerEncoder(acfg)
        self.semantic_tokenizer = TokenizerEncoder(scfg)
        self.acoustic_connector = SpeechConnector(acfg.get("vae_dim", 64), 3584)
        self.semantic_connector = SpeechConnector(scfg.get("vae_dim", 128), 3584)
        lc = config.get("decoder_config", {})
        from transformers import Qwen2Config, Qwen2ForCausalLM
        qc = Qwen2Config(hidden_size=lc.get("hidden_size", 3584), intermediate_size=lc.get("intermediate_size", 18944), num_hidden_layers=lc.get("num_hidden_layers", 28), num_attention_heads=lc.get("num_attention_heads", 28), num_key_value_heads=lc.get("num_key_value_heads", 4), vocab_size=lc.get("vocab_size", 152064), max_position_embeddings=lc.get("max_position_embeddings", 131072), rope_theta=lc.get("rope_theta", 1000000.0), use_cache=True, torch_dtype=str(dtype).replace("torch.", ""))
        self.language_model = Qwen2ForCausalLM(qc).model
        self.lm_head = nn.Linear(qc.hidden_size, qc.vocab_size, bias=False)
        self._audio_token_id = config.get("audio_token_id", 151648)
    def get_audio_features(self, waveform):
        with torch.no_grad():
            w = waveform.to(self._dtype)
            if w.dim() == 2: w = w.unsqueeze(1)
            a = self.acoustic_connector(self.acoustic_tokenizer(w).transpose(1, 2))
            s = self.semantic_connector(self.semantic_tokenizer(w).transpose(1, 2))
            return torch.cat([a, s], dim=1)
    def generate(self, input_ids, audio_embeds, max_new_tokens=512, eos_token_id=151643, temperature=0.0, **kw):
        device = input_ids.device
        embed = self.language_model.get_input_embeddings()
        te = embed(input_ids)
        ap = (input_ids == self._audio_token_id).nonzero(as_tuple=True)[1]
        if ap.numel() > 0:
            s, e = ap[0].item(), ap[-1].item() + 1
            ins = torch.cat([te[:,:s], audio_embeds.to(device, dtype=embed.weight.dtype), te[:,e:]], dim=1)
        else:
            ins = torch.cat([te, audio_embeds.to(device, dtype=embed.weight.dtype)], dim=1)
        attn = torch.ones(1, ins.shape[1], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.language_model(inputs_embeds=ins, attention_mask=attn, use_cache=True, return_dict=True)
            logits = self.lm_head(out.last_hidden_state[:,-1,:])
            token = logits.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(logits/temperature,-1),1).squeeze(-1)
            gen = [token.item()]
            pkv = out.past_key_values
            ce = embed(token.unsqueeze(0))
            for _ in range(max_new_tokens-1):
                if token.item() == eos_token_id: break
                out = self.language_model(inputs_embeds=ce, attention_mask=None, past_key_values=pkv, use_cache=True, return_dict=True)
                logits = self.lm_head(out.last_hidden_state[:,-1,:])
                token = logits.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(logits/temperature,-1),1).squeeze(-1)
                gen.append(token.item())
                pkv = out.past_key_values; ce = embed(token.unsqueeze(0))
        return torch.tensor([gen], device=device)
    @classmethod
    def from_pretrained(cls, model_path, dtype=torch.bfloat16):
        import json
        mp = Path(model_path)
        config = json.loads((mp / "config.json").read_text()) if (mp / "config.json").exists() else {}
        model = cls(config, dtype=dtype)
        state = {}
        for sf in sorted(mp.glob("*.safetensors")):
            from safetensors.torch import load_file
            state.update(load_file(str(sf), device="cpu"))
        model._load_weights(state)
        model = model.to(dtype) if dtype else model
        model._dtype = dtype or torch.float32
        model.eval()
        return model
    def _load_weights(self, state):
        clean = {}
        mk = set(self.state_dict().keys())
        for ck, t in state.items():
            k = ck.removeprefix("model.")
            for prefix in ("acoustic_tokenizer", "semantic_tokenizer"):
                if k.startswith(prefix + ".encoder."):
                    k = k.replace(prefix + ".encoder.", prefix + ".")
            if ".decoder." in k:
                continue
            if k in mk:
                clean[k] = t
        self.load_state_dict(clean, strict=False)
        n = len(clean); print(f"[VibeVoice] Loaded {n}/{len(mk)} keys")
    @property
    def device(self): return self._device
    @device.setter
    def device(self, d): self._device = d
