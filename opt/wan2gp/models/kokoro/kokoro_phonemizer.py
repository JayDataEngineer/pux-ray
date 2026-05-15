"""Kokoro phonemizer — espeak-ng via phonemizer library, no misaki/spacy."""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

LANG_MAP = {
    "a": "en-us",   # American English
    "b": "en-gb",   # British English
    "e": "es",       # Spanish
    "f": "fr-fr",    # French
    "h": "hi",       # Hindi
    "i": "it",       # Italian
    "p": "pt-br",    # Brazilian Portuguese
}

_espeak_cache: dict[str, object] = {}


def _get_espeak_backend(language: str):
    """Get or create espeak phonemizer backend for a language."""
    if language not in _espeak_cache:
        import phonemizer
        backend = phonemizer.backend.EspeakBackend(
            language=language,
            preserve_punctuation=True,
            with_stress=True,
        )
        _espeak_cache[language] = backend
    return _espeak_cache[language]


def phonemize(text: str, lang_code: str = "a") -> str:
    """Convert text to phonemes using espeak-ng.

    Args:
        text: Input text
        lang_code: Language code (a=en-us, b=en-gb, etc.)

    Returns:
        Phonemized string
    """
    text = text.strip()
    if not text:
        return ""

    language = LANG_MAP.get(lang_code)
    if not language:
        # Fallback: pass lang_code directly to espeak
        language = lang_code

    backend = _get_espeak_backend(language)
    ps = backend.phonemize([text])
    ps = ps[0] if ps else ""

    # Kokoro-specific phoneme post-processing
    ps = ps.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace(
        "kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ"
    )
    ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
    ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
    ps = re.sub(r" z(?=[;:,.!?¡¿—…\"«»\"\"]|$)", "z", ps)

    if language == "en-us":
        ps = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", ps)

    return ps.strip()


def chunk_phonemes(phonemes: str, max_len: int = 510) -> list[str]:
    """Split phoneme string into chunks at sentence boundaries.

    Tries to break at sentence-ending punctuation (.!?) first,
    then at clause boundaries (:;,), then just hard-splits.
    """
    if len(phonemes) <= max_len:
        return [phonemes]

    chunks = []
    remaining = phonemes
    while len(remaining) > max_len:
        # Try to find a sentence boundary within range
        best = -1
        for delim in [".", "!", "?", "…"]:
            pos = remaining[:max_len].rfind(delim)
            if pos > best:
                best = pos
        if best == -1:
            for delim in [":", ";", ","]:
                pos = remaining[:max_len].rfind(delim)
                if pos > best:
                    best = pos
        if best == -1:
            best = max_len

        chunk = remaining[:best + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[best + 1:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks
