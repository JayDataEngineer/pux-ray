"""Test MOSS-SoundEffect handler end-to-end against real model weights.

Usage: uv run python scripts/test_moss.py
"""
import sys
import os
import time
import json
import base64
import struct

# Add the wan2gp models path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "opt", "wan2gp")))

MODELS_ROOT = os.environ.get("TECH_NOIR_MODELS_ROOT", "/home/user/Documents/models")
MOSS_PATH = os.path.join(MODELS_ROOT, "audio", "moss-soundeffect")
TOKENIZER_PATH = os.path.join(MODELS_ROOT, "audio", "moss-audio-tokenizer")


def test_weight_loading():
    """Test that handler loads weights correctly."""
    import torch
    import safetensors.torch
    from pathlib import Path
    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3Model

    mp = Path(MOSS_PATH)
    print(f"1. Loading config from {mp}")
    hf_config = AutoConfig.from_pretrained(
        str(mp), trust_remote_code=True, local_files_only=True,
    )
    print(f"   Config: n_vq={hf_config.n_vq}, audio_vocab_size={hf_config.audio_vocab_size}, "
          f"sampling_rate={hf_config.sampling_rate}")
    print(f"   language_config: model_type={hf_config.language_config.model_type}, "
          f"hidden_size={hf_config.language_config.hidden_size}, "
          f"vocab_size={hf_config.language_config.vocab_size}, "
          f"num_layers={hf_config.language_config.num_hidden_layers}")

    # Load language model with correct Qwen3 class
    lang_cfg = hf_config.language_config
    print(f"\n2. Creating Qwen3Model from config")
    lang_model = Qwen3Model(lang_cfg)
    print(f"   Model params: {sum(p.numel() for p in lang_model.parameters()) / 1e6:.1f}M")

    # Load safetensors
    print(f"\n3. Loading safetensors from {mp}")
    sd = {}
    for sf_path in sorted(mp.rglob("model*.safetensors")):
        chunk = safetensors.torch.load_file(str(sf_path))
        sd.update(chunk)
        print(f"   {sf_path.name}: {len(chunk)} keys")
    print(f"   Total keys: {len(sd)}")

    # Check language model weight overlap
    lang_prefix = "language_model."
    lang_sd = {}
    for k, v in sd.items():
        if k.startswith(lang_prefix):
            lang_sd[k[len(lang_prefix):]] = v.to(dtype=torch.bfloat16)

    lang_state = lang_model.state_dict()
    matched = set(lang_sd.keys()) & set(lang_state.keys())
    missing = set(lang_state.keys()) - set(lang_sd.keys())
    extra = set(lang_sd.keys()) - set(lang_state.keys())
    print(f"\n4. Language model weight matching:")
    print(f"   Matched: {len(matched)}/{len(lang_state)}")
    if missing:
        print(f"   Missing (first 5): {sorted(missing)[:5]}")
    if extra:
        print(f"   Extra (first 5): {sorted(extra)[:5]}")

    # Check emb_ext keys
    n_vq = hf_config.n_vq
    audio_vocab = hf_config.audio_vocab_size
    hidden = lang_cfg.hidden_size
    text_vocab = lang_cfg.vocab_size
    print(f"\n5. Checking emb_ext (n_vq={n_vq}, audio_vocab={audio_vocab}):")
    for i in range(n_vq):
        key = f"emb_ext.{i}.weight"
        if key in sd:
            print(f"   {key}: shape={sd[key].shape}")
        else:
            print(f"   {key}: MISSING")

    # Check lm_heads keys
    print(f"\n6. Checking lm_heads (text_vocab={text_vocab}):")
    for i in range(n_vq + 1):
        key = f"lm_heads.{i}.weight"
        if key in sd:
            print(f"   {key}: shape={sd[i].shape}")
        else:
            print(f"   {key}: MISSING")

    return hf_config, lang_model, sd, lang_sd


def test_full_handler():
    """Test the actual family_handler.load_model + generate."""
    import torch
    from models.moss.moss_handler import family_handler

    model_def = {
        "moss_soundeffect_path": MOSS_PATH,
        "moss_audio_tokenizer_path": TOKENIZER_PATH,
    }

    print(f"\n=== Testing family_handler.load_model ===")
    t0 = time.time()
    pipeline, result = family_handler.load_model(
        model_filename="moss-soundeffect",
        model_type="moss-soundeffect",
        base_model_type="moss-soundeffect",
        model_def=model_def,
        dtype=torch.bfloat16,
    )
    print(f"   Loaded in {time.time()-t0:.1f}s")
    print(f"   Pipe keys: {list(result['pipe'].keys())}")
    print(f"   Co-tenants: {result['coTenantsMap']}")

    for k, v in result['pipe'].items():
        if v is not None and hasattr(v, 'parameters'):
            n = sum(p.numel() for p in v.parameters())
            print(f"   {k}: {n/1e6:.1f}M params, dtype={next(v.parameters()).dtype}")
        else:
            print(f"   {k}: {type(v).__name__}")

    print(f"\n=== Testing generate ===")
    t0 = time.time()
    out = pipeline.generate(input_prompt="gentle rain on a tin roof")
    elapsed = time.time() - t0
    print(f"   Generated in {elapsed:.1f}s")
    print(f"   Status: {out.get('status')}")
    print(f"   Media type: {out.get('media_type')}")

    if out.get("data"):
        raw = base64.b64decode(out["data"])
        print(f"   Output size: {len(raw)} bytes")
        # Check if WAV is silence or actual audio
        if len(raw) > 44:
            # Parse WAV to check if it's all zeros
            samples = struct.unpack(f'<{min(16000, (len(raw)-44)//2)}h', raw[44:44+32000])
            max_val = max(abs(s) for s in samples)
            print(f"   Max sample amplitude: {max_val} ({'SILENCE' if max_val < 10 else 'HAS AUDIO'})")
    else:
        print(f"   No data returned!")
        if out.get("text"):
            print(f"   Text: {out['text'][:200]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-only", action="store_true", help="Only test weight loading, not generate")
    args = parser.parse_args()

    if args.weights_only:
        test_weight_loading()
    else:
        test_full_handler()
