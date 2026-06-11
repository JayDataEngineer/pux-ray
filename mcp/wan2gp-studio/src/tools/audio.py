"""Audio tools — speech recognition, sound effects, and music generation.

- transcribe: Speech-to-text (faster_whisper CPU, vibevoice_asr GPU)
- generate_sound: Text-to-sound-effect (MOSS-SoundEffect 8B GPU)
- generate_music: Text-to-music (ACE-Step 1.5 GPU)
- voice_creator_examples: Get voice creation examples from vendor demos
"""
from __future__ import annotations

import base64
from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field

# Import voice examples
from .voice_examples import (
    VOICE_EXAMPLES,
    VOICE_CATEGORIES,
    MOSS_SAMPLING_PRESETS,
    QWEN3_VOICE_PRESETS,
    QWEN3_MODE_DESCRIPTIONS,
    PAUSE_CONTROL_EXAMPLES,
    DIALOGUE_EXAMPLES,
    get_example_by_id,
    get_examples_by_language,
    get_examples_by_category,
)


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
    # Qwen3-TTS mode control
    mode: Annotated[str, Field(
        description="Qwen3-TTS mode: custom_voice (preset), voice_design (describe), voice_clone (from audio).",
        enum=["custom_voice", "voice_design", "voice_clone"],
    )] = "voice_design",
    voice: Annotated[str | None, Field(
        description="Voice preset name for Qwen3-TTS custom_voice mode (Aiden, Chloe, Ethan, etc.).",
    )] = None,
    instruct: Annotated[str | None, Field(
        description="Voice description. E.g. 'warm female voice with a southern accent'.",
    )] = None,
    ref_audio_b64: Annotated[str | None, Field(
        description="Base64-encoded reference audio for voice cloning. Can upload multiple for better cloning.",
    )] = None,
    ref_audio_b64_list: Annotated[list[str] | None, Field(
        description="[ADVANCED] Multiple reference audio samples for improved voice cloning.",
    )] = None,
    language: Annotated[str, Field(
        description="Language.",
        enum=["English", "Chinese", "Japanese", "Korean"],
    )] = "English",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    # ── MOSS VoiceGenerator sampling params (1:1 with vendor demo) ──
    max_new_tokens: Annotated[int, Field(
        description="[MOSS] Maximum tokens to generate. Higher = longer audio. 4096 default.",
    )] = 4096,
    audio_temperature: Annotated[float, Field(
        description="[MOSS] Audio sampling temperature. Higher = more variation. 1.5 recommended.",
    )] = 1.5,
    audio_top_p: Annotated[float, Field(
        description="[MOSS] Audio nucleus sampling threshold. 0.6 recommended.",
    )] = 0.6,
    audio_top_k: Annotated[int, Field(
        description="[MOSS] Audio top-k sampling. 50 recommended.",
    )] = 50,
    audio_repetition_penalty: Annotated[float, Field(
        description="[MOSS] Audio repetition penalty. 1.1 recommended.",
    )] = 1.1,
    # ── Model variant selection ──
    model_variant: Annotated[str, Field(
        description="[MOSS] Model variant: default, nano (faster), or v1.5 (latest features).",
        enum=["default", "nano", "v1.5"],
    )] = "default",
    # ── Pause control (MOSS-TTS v1.5) ──
    enable_pauses: Annotated[bool, Field(
        description="[MOSS v1.5] Enable [pause X.Xs] markers in text for timing control.",
    )] = False,
    ctx: Context | None = None,
) -> dict:
    """Create a custom voice using voice design or voice cloning.

    Generates an audio sample with the designed voice. The output can be
    saved as an asset and used as a reference voice in Text to Speech.

    moss_voicegenerator: Describe or clone a voice on GPU with full sampling control.
    qwen3_tts: Use Qwen3-TTS custom_voice/voice_design/voice_clone modes.

    Vendor-recommended MOSS settings: audio_temperature=1.5, audio_top_p=0.6,
    audio_top_k=50, audio_repetition_penalty=1.1

    Pause Control: Use [pause 1.5s] syntax in text for timing (MOSS v1.5 only).
    Multiple References: Provide ref_audio_b64_list for better voice cloning.
    """
    forge = _forge(ctx)

    if engine == "qwen3_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "faster_qwen3_tts",
            "text": text,
            "language": language,
        }

        # Handle mode selection for Qwen3-TTS
        if mode == "voice_clone" and (ref_audio_b64 or ref_audio_b64_list):
            payload["mode"] = "voice_clone"
            # Use multiple references if provided
            if ref_audio_b64_list and len(ref_audio_b64_list) > 0:
                payload["ref_audio_b64"] = ref_audio_b64_list[0]  # Primary reference
                if len(ref_audio_b64_list) > 1:
                    payload["ref_audio_b64_list"] = ref_audio_b64_list
            else:
                payload["ref_audio_b64"] = ref_audio_b64
        elif mode == "voice_design":
            payload["mode"] = "voice_design"
            payload["instruct"] = instruct or ""
        elif mode == "custom_voice":
            payload["mode"] = "custom_voice"
            payload["voice"] = voice or "Aiden"
        else:
            # Fallback to voice design if mode doesn't match
            payload["mode"] = "voice_design"
            payload["instruct"] = instruct or ""
    else:
        # moss_voicegenerator with full sampling control
        model_name = "moss-voicegenerator"
        if model_variant == "nano":
            model_name = "moss-tts-nano"
        elif model_variant == "v1.5":
            model_name = "moss-tts-v1.5"

        payload = {
            "service": "wan2gp",
            "model": model_name,
            "text": text,
            "language": language,
            # Add MOSS sampling parameters matching vendor demo
            "max_new_tokens": max_new_tokens,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
            # Pause control
            "enable_pauses": enable_pauses,
        }
        if instruct:
            payload["instruction"] = instruct
        if ref_audio_b64 or ref_audio_b64_list:
            # Use multiple references if provided
            if ref_audio_b64_list and len(ref_audio_b64_list) > 0:
                payload["ref_audio_b64"] = ref_audio_b64_list[0]  # Primary reference
                if len(ref_audio_b64_list) > 1:
                    payload["ref_audio_b64_list"] = ref_audio_b64_list
            else:
                payload["ref_audio_b64"] = ref_audio_b64

    if seed >= 0:
        payload["seed"] = seed

    return await forge.invoke(payload)


async def voice_creator_examples(
    example_id: Annotated[str | None, Field(
        description="Specific example ID to retrieve (e.g., 'zh/0', 'en/1'). If null, returns all examples.",
    )] = None,
    language: Annotated[str | None, Field(
        description="Filter examples by language ('Chinese', 'English').",
    )] = None,
    category: Annotated[str | None, Field(
        description="Filter examples by category (emotional, character_change, age_voice, etc.).",
    )] = None,
) -> dict:
    """Get voice creation examples from vendor demos.

    Returns curated voice creation examples matching the MOSS-VoiceGenerator vendor demo.
    These examples show how to write effective voice instructions for different emotions,
    character types, and vocal styles.

    Use these examples as templates for creating your own custom voices.
    """
    if example_id:
        example = get_example_by_id(example_id)
        if not example:
            return {
                "status": "error",
                "error": f"Example '{example_id}' not found",
            }
        return {
            "status": "ok",
            "examples": [example],
            "categories": VOICE_CATEGORIES,
            "sampling_presets": MOSS_SAMPLING_PRESETS,
            "qwen3_voices": QWEN3_VOICE_PRESETS,
            "qwen3_modes": QWEN3_MODE_DESCRIPTIONS,
            "pause_control_examples": PAUSE_CONTROL_EXAMPLES,
            "dialogue_examples": DIALOGUE_EXAMPLES,
        }

    # Filter examples if requested
    examples = VOICE_EXAMPLES
    if language:
        examples = get_examples_by_language(language)
    if category:
        examples = get_examples_by_category(category)

    return {
        "status": "ok",
        "examples": examples,
        "categories": VOICE_CATEGORIES,
        "sampling_presets": MOSS_SAMPLING_PRESETS,
        "qwen3_voices": QWEN3_VOICE_PRESETS,
        "qwen3_modes": QWEN3_MODE_DESCRIPTIONS,
        "pause_control_examples": PAUSE_CONTROL_EXAMPLES,
        "dialogue_examples": DIALOGUE_EXAMPLES,
    }


async def voice_creator_batch(
    requests: Annotated[list[dict], Field(
        description="List of voice generation requests. Each request should contain: text, engine, instruct, etc.",
    )],
    ctx: Context | None = None,
) -> dict:
    """Create multiple custom voices in batch.

    Processes multiple voice generation requests efficiently. Each request
    can have different instructions, engines, and parameters.

    Example:
    ```
    requests = [
        {"text": "Hello", "engine": "moss_voicegenerator", "instruct": "warm female voice"},
        {"text": "Hi there", "engine": "qwen3_tts", "mode": "voice_design", "instruct": "energetic male"},
    ]
    ```
    """
    forge = _forge(ctx)
    results = []

    for i, req in enumerate(requests):
        try:
            # Extract parameters with defaults
            text = req.get("text", "")
            engine = req.get("engine", "moss_voicegenerator")
            mode = req.get("mode", "voice_design")
            voice = req.get("voice")
            instruct = req.get("instruct")
            ref_audio_b64 = req.get("ref_audio_b64")
            ref_audio_b64_list = req.get("ref_audio_b64_list")
            language = req.get("language", "English")
            seed = req.get("seed", -1)
            max_new_tokens = req.get("max_new_tokens", 4096)
            audio_temperature = req.get("audio_temperature", 1.5)
            audio_top_p = req.get("audio_top_p", 0.6)
            audio_top_k = req.get("audio_top_k", 50)
            audio_repetition_penalty = req.get("audio_repetition_penalty", 1.1)
            model_variant = req.get("model_variant", "default")
            enable_pauses = req.get("enable_pauses", False)

            # Build payload based on engine
            if engine == "qwen3_tts":
                payload = {
                    "service": "wan2gp",
                    "model": "faster_qwen3_tts",
                    "text": text,
                    "language": language,
                }

                if mode == "voice_clone" and (ref_audio_b64 or ref_audio_b64_list):
                    payload["mode"] = "voice_clone"
                    if ref_audio_b64_list and len(ref_audio_b64_list) > 0:
                        payload["ref_audio_b64"] = ref_audio_b64_list[0]
                    else:
                        payload["ref_audio_b64"] = ref_audio_b64
                elif mode == "voice_design":
                    payload["mode"] = "voice_design"
                    payload["instruct"] = instruct or ""
                elif mode == "custom_voice":
                    payload["mode"] = "custom_voice"
                    payload["voice"] = voice or "Aiden"
            else:
                # moss_voicegenerator
                model_name = "moss-voicegenerator"
                if model_variant == "nano":
                    model_name = "moss-tts-nano"
                elif model_variant == "v1.5":
                    model_name = "moss-tts-v1.5"

                payload = {
                    "service": "wan2gp",
                    "model": model_name,
                    "text": text,
                    "language": language,
                    "max_new_tokens": max_new_tokens,
                    "audio_temperature": audio_temperature,
                    "audio_top_p": audio_top_p,
                    "audio_top_k": audio_top_k,
                    "audio_repetition_penalty": audio_repetition_penalty,
                    "enable_pauses": enable_pauses,
                }
                if instruct:
                    payload["instruction"] = instruct
                if ref_audio_b64:
                    payload["ref_audio_b64"] = ref_audio_b64

            if seed >= 0:
                payload["seed"] = seed

            result = await forge.invoke(payload)
            results.append({
                "index": i,
                "status": result.get("status", "unknown"),
                "data": result.get("data"),
                "media_type": result.get("media_type"),
                "error": result.get("error"),
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
            })

    return {
        "status": "ok",
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }


async def generate_batch(
    requests: Annotated[list[dict], Field(
        description="List of generation requests. Each request should contain: prompt, model, and optional parameters.",
    )],
    ctx: Context | None = None,
) -> dict:
    """Generate multiple images in batch.

    Processes multiple image generation requests efficiently. Each request
    can have different prompts, models, and parameters.

    Example:
    ```
    requests = [
        {"prompt": "a sunset over mountains", "model": "anima_base"},
        {"prompt": "a portrait of a person", "model": "z_image"},
    ]
    ```
    """
    forge = _forge(ctx)
    results = []

    for i, req in enumerate(requests):
        try:
            # Extract parameters
            prompt = req.get("prompt", req.get("text", ""))
            model = req.get("model", "anima_base")
            width = req.get("width", 1024)
            height = req.get("height", 1024)
            steps = req.get("sampling_steps", 30)
            guidance = req.get("guide_scale", 4.0)
            seed = req.get("seed", -1)
            negative_prompt = req.get("negative_prompt")

            payload = {
                "service": "wan2gp",
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "sampling_steps": steps,
                "guide_scale": guidance,
            }
            if seed >= 0:
                payload["seed"] = seed
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt

            result = await forge.invoke(payload)
            results.append({
                "index": i,
                "status": result.get("status", "unknown"),
                "data": result.get("data"),
                "media_type": result.get("media_type"),
                "error": result.get("error"),
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
            })

    return {
        "status": "ok",
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }


async def generate_music_batch(
    requests: Annotated[list[dict], Field(
        description="List of music generation requests. Each request should contain: prompt, duration, etc.",
    )],
    ctx: Context | None = None,
) -> dict:
    """Generate multiple music tracks in batch.

    Processes multiple music generation requests efficiently.
    """
    forge = _forge(ctx)
    results = []

    for i, req in enumerate(requests):
        try:
            prompt = req.get("prompt", "")
            duration_seconds = req.get("duration_seconds", 30.0)
            seed = req.get("seed")
            max_new_tokens = req.get("max_new_tokens", 4096)
            text_temperature = req.get("text_temperature", 1.0)
            text_top_p = req.get("text_top_p", 0.9)
            audio_temperature = req.get("audio_temperature", 1.0)
            audio_top_p = req.get("audio_top_p", 0.9)

            payload = {
                "service": "wan2gp",
                "model": "ace_step",
                "prompt": prompt,
                "duration": duration_seconds,
                "max_new_tokens": max_new_tokens,
                "text_temperature": text_temperature,
                "text_top_p": text_top_p,
                "audio_temperature": audio_temperature,
                "audio_top_p": audio_top_p,
            }
            if seed is not None:
                payload["seed"] = seed

            result = await forge.invoke(payload)
            results.append({
                "index": i,
                "status": result.get("status", "unknown"),
                "data": result.get("data"),
                "media_type": result.get("media_type"),
                "error": result.get("error"),
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
            })

    return {
        "status": "ok",
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }


async def generate_sound_batch(
    requests: Annotated[list[dict], Field(
        description="List of sound effect generation requests.",
    )],
    ctx: Context | None = None,
) -> dict:
    """Generate multiple sound effects in batch.

    Processes multiple sound effect generation requests efficiently.
    """
    forge = _forge(ctx)
    results = []

    for i, req in enumerate(requests):
        try:
            prompt = req.get("prompt", "")
            duration_seconds = req.get("duration_seconds", 5.0)
            seed = req.get("seed")
            max_new_tokens = req.get("max_new_tokens", 4096)

            payload = {
                "service": "wan2gp",
                "model": "moss/moss-soundeffect",
                "prompt": prompt,
                "duration": duration_seconds,
                "max_new_tokens": max_new_tokens,
            }
            if seed is not None:
                payload["seed"] = seed

            result = await forge.invoke(payload)
            results.append({
                "index": i,
                "status": result.get("status", "unknown"),
                "data": result.get("data"),
                "media_type": result.get("media_type"),
                "error": result.get("error"),
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
            })

    return {
        "status": "ok",
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }
