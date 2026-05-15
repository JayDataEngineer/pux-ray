"""Emotion presets for Qwen3-TTS instruct-based emotion control.

Each preset maps an emotion name to an English/Chinese instruct string
that the Qwen3-TTS CustomVoice model understands.

Usage:
    from qwen_tts.emotions import EMOTION_PRESETS, emotion_to_instruct

    instruct = emotion_to_instruct("happy")
    # → "Speak in a cheerful and happy tone, voice light and bright"
"""

EMOTION_PRESETS: dict[str, str] = {
    "happy": "Speak in a cheerful and happy tone, voice light and bright",
    "sad": "Speak sadly, voice slightly trembling, slower pace",
    "angry": "Speak with suppressed anger, voice low, suddenly raising at the end",
    "scared": "Speak with fear and trembling, getting faster as you go",
    "gentle": "Speak gently and softly, as if comforting someone",
    "tsundere": "Speak in a tsundere way, saying harsh things but unable to hide the embarrassment",
    "whisper": "Whisper softly, like telling a secret, breathy voice",
    "cold": "Speak in a cold and distant tone, no emotional fluctuation",
    "excited": "Speak with excitement and enthusiasm, energetic and upbeat",
    "shy": "Speak shyly, hesitant, soft voice with occasional pauses",
    "serious": "Speak seriously and firmly, clear and deliberate",
    "playful": "Speak playfully and teasingly, with a mischievous tone",
    "worried": "Speak with worry and concern, slightly anxious",
    "confident": "Speak with confidence and assurance, strong and clear",
    "tired": "Speak tiredly, slower pace, slightly quieter",
    "surprised": "Speak with genuine surprise, voice raised slightly",
    "neutral": "",  # No instruct = neutral voice
    "calm": "Speak calmly and steadily, with a soothing voice",
    "nervous": "Speak nervously with anxious energy, slightly stammering",
    "sarcastic": "Speak sarcastically with a dry, cutting tone",

    # VN character archetypes (Chinese, better for Qwen3-TTS)
    "cheerful": "用开朗愉快的语气说话，声音轻快明亮",
    "melancholy": "用忧郁悲伤的语气说话，声音微微颤抖",
    "fierce": "用凶狠强势的语气说话，声音低沉有力",
    "terrified": "用极度恐惧的语气说话，声音颤抖加速",
    "tender": "用温柔体贴的语气说话，像是在安抚对方",
    "cold_zh": "用冰冷疏离的语气说话，毫无感情波动",
    "mysterious": "用神秘低语的方式说话，欲言又止",
    "arrogant": "用傲慢轻蔑的语气说话，居高临下",
    "desperate": "用绝望急切的语气说话，声音带着哭腔",
    "heroic": "用坚定英勇的语气说话，声音洪亮有力",
    "scheming": "用算计腹黑的语气说话，语气意味深长",
    "innocent": "用天真无辜的语气说话，声音甜美单纯",
    "drunken": "用醉酒迷糊的语气说话，吐字不清语速缓慢",
}


def emotion_to_instruct(emotion: str) -> str | None:
    """Convert an emotion name to its instruct string.

    Returns None for "neutral" (no instruct = neutral voice).
    Returns the emotion name as-is if it's not a known preset
    (treated as a raw instruct string).
    """
    if emotion.lower() == "neutral":
        return None
    known = EMOTION_PRESETS.get(emotion.lower())
    if known is not None:
        return known if known else None  # empty string = neutral
    # Pass through as raw instruct string
    return emotion
