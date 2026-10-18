#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Policies deciding whether an eager transcript still matches the committed one."""

import re
import unicodedata
from abc import ABC, abstractmethod

# Apostrophes are dropped rather than spaced out, so a service that writes
# "it's" matches one that writes "its".
_APOSTROPHES = re.compile(r"['\u2019\u02bc]")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


class EagerMatchPolicy(ABC):
    """Decides whether a speculative response is still the right answer.

    A speculative response is generated from an eager transcript, before the
    service commits one. This decides whether the committed transcript is close
    enough to the eager one for that response to stand.
    """

    @abstractmethod
    def matches(self, eager: str, final: str) -> bool:
        """Report whether the committed transcript still matches the eager one.

        Args:
            eager: The transcript the speculative response was generated from.
            final: The transcript the service committed to.

        Returns:
            True to keep the speculative response, False to discard it and
            answer the committed transcript instead.
        """


class ExactMatch(EagerMatchPolicy):
    """Keep the response only when the two transcripts are identical."""

    def matches(self, eager: str, final: str) -> bool:
        """Report whether the two transcripts are identical."""
        return eager == final


class NormalizedMatch(EagerMatchPolicy):
    """Keep the response when the transcripts differ only in formatting.

    Compares with case, punctuation and whitespace removed. Services commonly
    format the committed transcript — capitalizing it, punctuating it — while
    leaving the eager one raw, which :class:`ExactMatch` counts as a difference.
    """

    def matches(self, eager: str, final: str) -> bool:
        """Report whether the two transcripts match once formatting is removed."""
        return self._normalize(eager) == self._normalize(final)

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).casefold()
        text = _APOSTROPHES.sub("", text)
        text = _PUNCTUATION.sub(" ", text)
        return _WHITESPACE.sub(" ", text).strip()
