#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replace words with a TTS service's pronunciation markup.

Applications usually reach this through a service's
:meth:`~pipecat.services.tts_service.TTSService.pronunciation_transform_ipa`
classmethod, which supplies the service's own formatter.
"""

import re
from collections.abc import Callable, Mapping

from loguru import logger

from pipecat.frames.frames import AggregationType

PronunciationFormatter = Callable[[str, str], str | None]


class PronunciationTransform:
    """A text transform that replaces words with a service's pronunciation markup.

    TTS services recognize it among their ``text_transforms`` and skip it while
    they cannot read pronunciation markup (see
    :attr:`~pipecat.services.tts_service.TTSService.supports_pronunciations`),
    so the words are spoken as written.
    """

    def __init__(self, pattern: re.Pattern | None, replace: Callable[[re.Match], str]):
        """Initialize the transform.

        Args:
            pattern: Matches the words to replace, or None when there are none.
            replace: Returns the text to send in place of a match.
        """
        self._pattern = pattern
        self._replace = replace

    async def __call__(self, text: str, aggregation_type: str | AggregationType) -> str:
        """Replace every matched word in ``text``.

        Args:
            text: The text about to be sent to the service.
            aggregation_type: The aggregation type of the text (unused).

        Returns:
            The text with each matched word replaced.
        """
        return self._pattern.sub(self._replace, text) if self._pattern else text


def pronunciation_transform(
    pronunciations: Mapping[str, str],
    format_pronunciation: PronunciationFormatter,
    *,
    service_name: str = "TTS service",
) -> PronunciationTransform:
    """Return a transform that replaces each word with its formatted pronunciation.

    Words match whole and case-insensitively, longest first, and never as part of
    a longer or hyphenated word. The matched text is passed to the formatter, so
    markup that keeps the word (such as an SSML ``<phoneme>`` tag) keeps it as
    written.

    Every pronunciation is formatted once here, so one the service cannot use is
    reported when the transform is built rather than while speaking. Those words
    are left out and spoken as written.

    Args:
        pronunciations: Word to IPA, e.g. ``{"Metformin": "mɛtˈfɔɹmɪn"}``.
        format_pronunciation: Turns ``(word, ipa)`` into the text to send, or None
            when it cannot.
        service_name: Name used in the warning for unusable pronunciations.

    Returns:
        A transform to register with ``text_transforms``.
    """
    phonemes: dict[str, str] = {}
    unusable: list[str] = []
    for word, value in pronunciations.items():
        word = word.strip()
        if not word:
            continue
        if format_pronunciation(word, value) is None:
            unusable.append(word)
        else:
            phonemes[word.lower()] = value

    if unusable:
        logger.warning(
            f"{service_name} cannot use these IPA pronunciations, so they "
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
        return format_pronunciation(word, phonemes[word.lower()]) or word

    return PronunciationTransform(pattern, _replace)
