"""Voice creation examples and presets matching vendor demos.

These examples are sourced from the MOSS-VoiceGenerator vendor demo:
https://github.com/OpenMOSS/MOSS-TSS/tree/main/moss_tts_realtime
"""
from __future__ import annotations

# Voice creation examples matching vendor demo format
VOICE_EXAMPLES = [
    # Chinese examples from vendor demo
    {
        "id": "zh/0",
        "language": "Chinese",
        "instruction": "撕心裂肺，声泪俱下的中年女性",
        "text": "皇上，臣妾做不到啊！皇上，您就杀了臣妾吧！",
        "category": "emotional"
    },
    {
        "id": "zh/1",
        "language": "Chinese",
        "instruction": "年轻女性，开头傲慢不屑，发现对方身份后秒怂，疯狂道歉，惊慌失措",
        "text": "你谁啊，关你什么事？啊…王总，您好您好，我不知道是您……",
        "category": "character_change"
    },
    {
        "id": "zh/2",
        "language": "Chinese",
        "instruction": "疲惫沙哑的老年声音缓慢抱怨，带有轻微呻吟。",
        "text": "哎呀，我的老腰啊，这年纪大了就是不行了。",
        "category": "age_voice"
    },
    {
        "id": "zh/3",
        "language": "Chinese",
        "instruction": "粗犷急躁的海盗船长，语速快，语调低沉而充满命令，带着一股不容置疑的霸道。",
        "text": "快点！把那箱金币搬过来！速度快点！别磨磨蹭蹭的！我们必须在涨潮之前离开这里，否则就来不及了！",
        "category": "character"
    },
    # English examples from vendor demo
    {
        "id": "en/0",
        "language": "English",
        "instruction": "Mom scolding kid for breaking a vase, then seeing he cut himself, shifting to concern",
        "text": "How many times have I told you not to run in the house?! You could have…… oh honey, you're bleeding! Let me see your hand…… It's okay, baby.",
        "category": "emotional_shift"
    },
    {
        "id": "en/1",
        "language": "English",
        "instruction": "An elderly female voice, slightly nasal and soft, speaking in a frail, polite British tone, conveying subtle discomfort with gentle hesitation.",
        "text": "Achoo! Oh dear, I do believe I'm catching a cold. This dreadful weather is just too much.",
        "category": "age_voice"
    },
    {
        "id": "en/2",
        "language": "English",
        "instruction": "Little girl, innocent and curious, high-pitched and adorable",
        "text": "Mommy, why is the sky blue? And why do birds fly? And why-",
        "category": "age_voice"
    },
    {
        "id": "en/3",
        "language": "English",
        "instruction": "Emotional pop ballad with smooth, melodic delivery, slow tempo with gentle vibrato on sustained notes, conveying hope and vulnerability.",
        "text": "Walking down this empty street tonight, searching for a guiding light, stars above shine oh so bright, everything will be alright",
        "category": "singing"
    },
]

# Voice preset categories for easy filtering
VOICE_CATEGORIES = {
    "emotional": "Emotional expressions",
    "character_change": "Character/personality changes",
    "age_voice": "Age-specific voices",
    "character": "Character archetypes",
    "emotional_shift": "Emotional transitions",
    "singing": "Singing styles",
}

# Vendor-recommended sampling presets
MOSS_SAMPLING_PRESETS = {
    "default": {
        "audio_temperature": 1.5,
        "audio_top_p": 0.6,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    },
    "more_expressive": {
        "audio_temperature": 1.8,
        "audio_top_p": 0.7,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.0,
    },
    "more_stable": {
        "audio_temperature": 1.2,
        "audio_top_p": 0.5,
        "audio_top_k": 30,
        "audio_repetition_penalty": 1.2,
    },
    "creative": {
        "audio_temperature": 2.0,
        "audio_top_p": 0.8,
        "audio_top_k": 100,
        "audio_repetition_penalty": 1.0,
    },
}

# Qwen3-TTS voice presets
QWEN3_VOICE_PRESETS = [
    "Aiden", "Chloe", "Ethan", "Marcus", "Ono_Anna", "Sohee",
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
]

# Qwen3-TTS mode descriptions
QWEN3_MODE_DESCRIPTIONS = {
    "custom_voice": "Use preset voices (Aiden, Chloe, etc.)",
    "voice_design": "Describe a voice with detailed instructions",
    "voice_clone": "Clone from reference audio",
}

def get_example_by_id(example_id: str) -> dict | None:
    """Get an example by its ID (e.g., 'zh/0', 'en/1')."""
    for example in VOICE_EXAMPLES:
        if example["id"] == example_id:
            return example
    return None

def get_examples_by_language(language: str) -> list[dict]:
    """Get all examples for a specific language ('Chinese' or 'English')."""
    return [ex for ex in VOICE_EXAMPLES if ex["language"] == language]

def get_examples_by_category(category: str) -> list[dict]:
    """Get all examples for a specific category."""
    return [ex for ex in VOICE_EXAMPLES if ex.get("category") == category]


# Pause control examples (MOSS-TTS v1.5)
PAUSE_CONTROL_EXAMPLES = [
    {
        "name": "Dramatic Pause",
        "text": "I have something to tell you... [pause 2.0s] I'm leaving.",
        "description": "Use pause for dramatic effect before important statements",
    },
    {
        "name": "Natural Conversation",
        "text": "Let me think about that... [pause 1.0s] Yes, I agree completely.",
        "description": "Add natural thinking pauses between statements",
    },
    {
        "name": "Emotional Buildup",
        "text": "It was a dark and stormy night... [pause 0.5s] The wind was howling... [pause 0.5s] Thunder crashed overhead!",
        "description": "Use short pauses to build tension and emotion",
    },
]

# Dialogue script examples
DIALOGUE_EXAMPLES = [
    {
        "name": "Parent-Child Conversation",
        "script": """Mother: How was your day at school today?
Child: It was great! We learned about dinosaurs.
Mother: That sounds wonderful! What did you learn?
Child: T-Rex was the biggest and scariest one!""",
        "description": "Natural conversation between mother and child",
    },
    {
        "name": "Customer Service",
        "script": """Agent: Thank you for calling our support line. How can I help you?
Customer: Hi, I'm having trouble with my internet connection.
Agent: I'm sorry to hear that. Let me help you troubleshoot the issue.
Customer: That would be great. It's been really frustrating.""",
        "description": "Professional customer service interaction",
    },
    {
        "name": "News Interview",
        "script": """Interviewer: Welcome to our show. Tonight we have a special guest.
Guest: Thank you for having me. It's a pleasure to be here.
Interviewer: Can you tell us about your latest project?
Guest: Absolutely! It's been an incredible journey of discovery.""",
        "description": "Formal interview setting",
    },
]
