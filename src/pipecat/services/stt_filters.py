"""Shared STT filtering utilities.

Provides language-aware interruption word detection so that genuine user
interruptions (e.g. "perdon", "disculpa", "wait", "stop") are not
discarded by the "too short" echo-rejection filters in individual STT
providers.
"""

import logging

from pipecat.transcriptions.language import Language

logger = logging.getLogger(__name__)

# Words/phrases that signal a genuine user interruption, keyed by language
# prefix (e.g. "es", "en").  All entries are lowercase.
INTERRUPTION_WORDS = {
    "es": [
        "perdon", "perdón",
        "disculpa", "disculpe",
        "oye", "oiga",
        "espera", "espere",
        "para", "alto", "basta",
        "callate", "cállate",
        "silencio",
        "momento", "un momento",
        "escucha", "escuche",
        "por favor",
    ],
    "en": [
        "sorry", "excuse me",
        "hold on", "wait",
        "stop", "pause",
        "listen", "hang on",
        "one moment", "one second",
        "shut up", "quiet",
        "please",
    ],
    "pt": [
        "desculpa", "desculpe",
        "perdao", "perdão",
        "espera", "espere",
        "para",
        "momento", "um momento",
        "por favor",
    ],
    "fr": [
        "pardon",
        "excusez", "excuse",
        "attendez", "attends",
        "arrete", "arrête", "arrêtez",
        "stop",
        "un moment",
        "silence",
        "s'il vous plait", "s'il vous plaît",
    ],
}


def _language_prefix(language) -> str:
    """Extract the two-letter language prefix from a Language enum or string."""
    value = language.value if isinstance(language, Language) else str(language)
    return value.split("-")[0].lower()


def contains_interruption_word(transcript: str, language) -> bool:
    """Check if *transcript* contains a known interruption word for *language*.

    Returns True if any interruption word/phrase is found, meaning the
    transcript should NOT be discarded by short-text echo filters.
    """
    prefix = _language_prefix(language)
    words = INTERRUPTION_WORDS.get(prefix)
    if not words:
        return False

    text = transcript.strip().lower()
    for word in words:
        if word in text:
            logger.info(
                f"Interruption word detected ('{word}' in '{transcript}'), "
                "bypassing short-text filter"
            )
            return True
    return False
