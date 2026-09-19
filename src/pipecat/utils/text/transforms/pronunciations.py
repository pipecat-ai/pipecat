#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replace words with a TTS service's pronunciation markup.

Applications usually reach this through a service's classmethods,
:meth:`~pipecat.services.tts_service.TTSService.pronounce_ipa` and
:meth:`~pipecat.services.tts_service.TTSService.pronounce_arpabet`, which supply
the service's own formatter.
"""

import re
from collections.abc import Awaitable, Callable, Mapping

from loguru import logger

from pipecat.frames.frames import AggregationType
from pipecat.utils.text.phonemes import PhonemeAlphabet

PronunciationFormatter = Callable[[str, str, PhonemeAlphabet], str | None]


def pronunciation_transform(
    pronunciations: Mapping[str, str],
    alphabet: PhonemeAlphabet,
    format_pronunciation: PronunciationFormatter,
    *,
    service_name: str = "TTS service",
) -> Callable[[str, str | AggregationType], Awaitable[str]]:
    """Return a transform that replaces each word with its formatted pronunciation.

    Words match whole and case-insensitively, longest first, and never as part of
    a longer or hyphenated word. The matched text is passed to the formatter, so
    markup that keeps the word (such as an SSML ``<phoneme>`` tag) keeps it as
    written.

    Every pronunciation is formatted once here, so one the service cannot use is
    reported when the transform is built rather than while speaking. Those words
    are left out and spoken as written.

    Args:
        pronunciations: Word to phoneme string, e.g. ``{"Metformin": "mɛtˈfɔɹmɪn"}``.
        alphabet: The alphabet every phoneme string is written in.
        format_pronunciation: Turns ``(word, phonemes, alphabet)`` into the text to
            send, or None when it cannot.
        service_name: Name used in the warning for unusable pronunciations.

    Returns:
        An async transform callable compatible with ``text_transforms``.
    """
    phonemes: dict[str, str] = {}
    unusable: list[str] = []
    for word, value in pronunciations.items():
        word = word.strip()
        if not word:
            continue
        if format_pronunciation(word, value, alphabet) is None:
            unusable.append(word)
        else:
            phonemes[word.lower()] = value

    if unusable:
        logger.warning(
            f"{service_name} cannot use these {alphabet.value} pronunciations, so they "
            f"will be spoken as written: {', '.join(unusable)}"
        )

    pattern = None
    if phonemes:
        words = sorted(phonemes, key=len, reverse=True)
        pattern = re.compile(
            r"(?<![\w-])(?:" + "|".join(re.escape(w) for w in words) + r")(?![\w-])",
            re.IGNORECASE,
        )

    def _replace(match: re.Match) -> str:
        word = match.group(0)
        return format_pronunciation(word, phonemes[word.lower()], alphabet) or word

    async def _transform(text: str, aggregation_type: str | AggregationType) -> str:
        return pattern.sub(_replace, text) if pattern else text

    return _transform
