"""Audio tools — speech recognition, sound effects, and music generation.

- transcribe: Speech-to-text (faster_whisper CPU, vibevoice_asr GPU)
- generate_sound: Text-to-sound-effect (MOSS-SoundEffect 8B GPU)
- generate_music: Text-to-music (ACE-Step 1.5 GPU)
"""
from __future__ import annotations

import base64
from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field


def _forge(ctx: Context) -> Any:
    fc = ctx.lifespan_context.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


async def transcribe(
    audio_b64: Annotated[str, Field(
        description="Base64-encoded audio file (wav, mp3, flac, ogg). "
                    "Required — provide the audio data to transcribe.",
    )],
    engine: Annotated[str, Field(
        description="ASR engine: 'whisper' (CPU, fast) or 'vibevoice' (GPU, diarization).",
    )] = "whisper",
    language: Annotated[str | None, Field(
        description="Language hint (e.g. 'en', 'ja', 'zh'). Auto-detects if omitted.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Transcribe speech from audio to text.

    Returns {status, text, language, segments?, speakers?}.
    Whisper is CPU-only and fast. VibeVoice adds speaker diarization on GPU.
    """
    forge = _forge(ctx)

    if engine == "vibevoice":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "vibevoice_asr",
            "audio_b64": audio_b64,
        }
        if language:
            payload["language"] = language
    else:
        payload = {
            "service": "wan2gp",
            "model": "faster_whisper",
            "audio_b64": audio_b64,
        }
        if language:
            payload["language"] = language

    return await forge.invoke(payload)


async def generate_sound(
    prompt: Annotated[str, Field(
        description="Description of the sound effect to generate. "
                    "E.g. 'thunder rumbling in the distance', 'sword unsheathing', "
                    "'crowd cheering in a stadium'.",
    )],
    duration_seconds: Annotated[float, Field(
        description="Target duration in seconds (1-30). Model may adjust slightly.",
    )] = 5.0,
    seed: Annotated[int | None, Field(
        description="Random seed for reproducibility.",
    )] = None,
    max_new_tokens: Annotated[int, Field(
        description="Maximum tokens to generate. Higher = longer audio. 4096 default.",
    )] = 4096,
    text_temperature: Annotated[float, Field(
        description="Text sampling temperature. Higher = more diverse. 1.0 default.",
    )] = 1.0,
    text_top_p: Annotated[float, Field(
        description="Text nucleus sampling threshold. 0.9 default.",
    )] = 0.9,
    text_top_k: Annotated[int, Field(
        description="Text top-k sampling. 50 default.",
    )] = 50,
    text_repetition_penalty: Annotated[float, Field(
        description="Text repetition penalty. 1.0 = disabled. Higher = less repetition.",
    )] = 1.0,
    audio_temperature: Annotated[float, Field(
        description="Audio sampling temperature. Higher = more diverse. 1.0 default.",
    )] = 1.0,
    audio_top_p: Annotated[float, Field(
        description="Audio nucleus sampling threshold. 0.9 default.",
    )] = 0.9,
    audio_top_k: Annotated[int, Field(
        description="Audio top-k sampling. 50 default.",
    )] = 50,
    audio_repetition_penalty: Annotated[float, Field(
        description="Audio repetition penalty. 1.0 = disabled. Higher = less repetition.",
    )] = 1.0,
    n_vq_for_inference: Annotated[int, Field(
        description="Number of VQ codebooks for inference. Fewer = faster, lower quality. 32 default.",
    )] = 32,
    ctx: Context | None = None,
) -> dict:
    """Generate a sound effect from a text description.

    Uses MOSS-SoundEffect 8B on GPU. Returns {status, data (base64 wav), media_type}.
    """
    forge = _forge(ctx)

    payload: dict[str, Any] = {
        "service": "wan2gp",
        "model": "moss/moss-soundeffect",
        "prompt": prompt,
        "duration": duration_seconds,
        "max_new_tokens": max_new_tokens,
        "text_temperature": text_temperature,
        "text_top_p": text_top_p,
        "text_top_k": text_top_k,
        "text_repetition_penalty": text_repetition_penalty,
        "audio_temperature": audio_temperature,
        "audio_top_p": audio_top_p,
        "audio_top_k": audio_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
        "n_vq_for_inference": n_vq_for_inference,
    }
    if seed is not None:
        payload["seed"] = seed

    return await forge.invoke(payload)


# Genre presets matching ACE-Step demo
GENRE_PRESETS = {
    "Custom": "",
    "Modern Pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant, female vocals, polished vocals",
    "Rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty, male vocals, raw vocals",
    "Hip Hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense, male vocals, rhythmic vocals",
    "Country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm, male vocals, twangy vocals",
    "EDM": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental",
    "Reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive, male vocals, smooth vocals",
    "Classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental",
    "Jazz": "jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful, male vocals, crooning vocals",
    "Metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy, male vocals, screamed vocals",
    "R&B": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic, female vocals, silky vocals",
}


async def generate_music(
    prompt: Annotated[str, Field(
        description="Music description / tags. E.g. 'funk, pop, soul, guitar, 105 BPM, energetic'. "
                    "Use commas to separate tags. Supports genre, instruments, tempo, mood, vocal style.",
    )],
    model: Annotated[str, Field(
        description="ACE-Step model variant.",
        enum=["v1.5", "v1.5 XL", "v1"],
    )] = "v1.5",
    lyrics: Annotated[str | None, Field(
        description="Lyrics with structure tags: [verse], [chorus], [bridge], [instrumental]. "
                    "Leave empty for instrumental.",
    )] = None,
    duration_seconds: Annotated[float, Field(
        description="Audio duration in seconds. -1 = random (30-240s).",
    )] = -1.0,
    format: Annotated[str, Field(
        description="Output audio format.",
        enum=["wav", "mp3", "ogg", "flac"],
    )] = "wav",
    genre_preset: Annotated[str, Field(
        description="Genre preset — auto-fills prompt tags. 'Custom' uses prompt as-is.",
        enum=list(GENRE_PRESETS.keys()),
    )] = "Custom",
    infer_step: Annotated[int, Field(
        description="Denoising steps. 30=fast, 60=balanced, 100+=high quality.",
    )] = 60,
    guidance_scale: Annotated[float, Field(
        description="CFG guidance scale. Higher = more prompt adherence. 15.0 is default.",
    )] = 15.0,
    guidance_scale_text: Annotated[float, Field(
        description="Separate guidance for text tags. Set >0 when using guidance_scale_lyric.",
    )] = 0.0,
    guidance_scale_lyric: Annotated[float, Field(
        description="Separate guidance for lyrics. Set >0 when using guidance_scale_text.",
    )] = 0.0,
    scheduler_type: Annotated[str, Field(
        description="Scheduler. euler=recommended, heun=slower/better, pingpong=SDE.",
        enum=["euler", "heun", "pingpong"],
    )] = "euler",
    cfg_type: Annotated[str, Field(
        description="CFG type. apg=recommended, cfg/cfg_star=classic.",
        enum=["cfg", "apg", "cfg_star"],
    )] = "apg",
    omega_scale: Annotated[float, Field(
        description="Granularity scale. Higher reduces artifacts. 10.0 default.",
    )] = 10.0,
    guidance_interval: Annotated[float, Field(
        description="Apply guidance only in this fraction of steps. 0.5 = middle half.",
    )] = 0.5,
    guidance_interval_decay: Annotated[float, Field(
        description="Decay guidance_scale to min_guidance_scale within interval. 0=no decay.",
    )] = 0.0,
    min_guidance_scale: Annotated[float, Field(
        description="End scale for guidance interval decay.",
    )] = 3.0,
    use_erg_tag: Annotated[bool, Field(
        description="Entropy Rectifying Guidance for tags — improves diversity.",
    )] = True,
    use_erg_lyric: Annotated[bool, Field(
        description="ERG for lyric encoder attention.",
    )] = False,
    use_erg_diffusion: Annotated[bool, Field(
        description="ERG for diffusion model attention.",
    )] = True,
    seed: Annotated[int, Field(
        description="Random seed. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Generate music from a text description.

    Full ACE-Step pipeline with all parameters matching the official demo.
    Supports v1, v1.5, and v1.5 XL models.
    Returns {status, data (base64 audio), media_type}.
    """
    forge = _forge(ctx)

    model_map = {
        "v1.5": "tts/ace_step_v1_5",
        "v1.5 XL": "tts/ace_step_v1_5_xl",
        "v1": "tts/ace_step_v1",
    }

    # Apply genre preset if not Custom
    actual_prompt = prompt
    if genre_preset != "Custom" and GENRE_PRESETS.get(genre_preset):
        actual_prompt = GENRE_PRESETS[genre_preset]

    payload: dict[str, Any] = {
        "service": "wan2gp",
        "model": model_map.get(model, "tts/ace_step_v1_5"),
        "prompt": actual_prompt,
        "lyrics": lyrics or "",
        "duration": duration_seconds,
        "format": format,
        "sampling_steps": infer_step,
        "guide_scale": guidance_scale,
        "guide_scale_text": guidance_scale_text,
        "guide_scale_lyric": guidance_scale_lyric,
        "scheduler_type": scheduler_type,
        "cfg_type": cfg_type,
        "omega_scale": omega_scale,
        "guidance_interval": guidance_interval,
        "guidance_interval_decay": guidance_interval_decay,
        "min_guidance_scale": min_guidance_scale,
        "use_erg_tag": use_erg_tag,
        "use_erg_lyric": use_erg_lyric,
        "use_erg_diffusion": use_erg_diffusion,
    }
    if seed >= 0:
        payload["seed"] = seed

    return await forge.invoke(payload)


async def voice_creator(
    text: Annotated[str, Field(
        description="Sample text to speak. Used to audition the generated voice.",
    )],
    engine: Annotated[str, Field(
        description="Voice creation engine.",
        enum=["moss_voicegenerator", "qwen3_tts"],
    )] = "moss_voicegenerator",
    instruct: Annotated[str | None, Field(
        description="Voice description. E.g. 'warm female voice with a southern accent'.",
    )] = None,
    ref_audio_b64: Annotated[str | None, Field(
        description="Base64-encoded reference audio for voice cloning.",
    )] = None,
    language: Annotated[str, Field(
        description="Language.",
        enum=["English", "Chinese", "Japanese", "Korean"],
    )] = "English",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Create a custom voice using voice design or voice cloning.

    Generates an audio sample with the designed voice. The output can be
    saved as an asset and used as a reference voice in Text to Speech.

    moss_voicegenerator: Describe or clone a voice on GPU.
    qwen3_tts: Use Qwen3-TTS voice_design/voice_clone mode.
    """
    forge = _forge(ctx)

    if engine == "qwen3_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "faster_qwen3_tts",
            "text": text,
            "language": language,
        }
        if ref_audio_b64:
            payload["mode"] = "voice_clone"
            payload["ref_audio_b64"] = ref_audio_b64
        else:
            payload["mode"] = "voice_design"
            payload["instruct"] = instruct or ""
    else:
        # moss_voicegenerator
        payload = {
            "service": "wan2gp",
            "model": "moss-voicegenerator",
            "text": text,
            "language": language,
        }
        if instruct:
            payload["instruct"] = instruct
        if ref_audio_b64:
            payload["ref_audio_b64"] = ref_audio_b64

    if seed >= 0:
        payload["seed"] = seed

    return await forge.invoke(payload)
