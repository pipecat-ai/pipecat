#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The expression of the words a TTS service reports speaking.

Part of expressive conversation (alpha). A TTS service reports the words it
speaks with the stretch of the LLM's text each covers, tags included.
:class:`SpokenExpression` keeps the text sent in one audio context, with what
each stretch of it performs, so each reported word can be given the modes it
is spoken in and the events performed before it.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass

from pipecat.expressive.renderer import ExpressionSpan
from pipecat.frames.frames import TextType


@dataclass
class _Span:
    """What a stretch of the text performs: words in their modes, or an event."""

    start: int
    end: int
    modes: list[str] | None = None
    event: tuple[TextType, str] | None = None
    # Whether the event has been reported, as by markup the service reported.
    reported: bool = False


class SpokenExpression:
    """Tells the modes and events of the words a TTS service reports speaking in one audio context."""

    def __init__(self):
        """Initialize the spoken expression."""
        self._text = ""
        self._spans: list[_Span] = []
        # How far into the text the service has reported speaking.
        self._position = 0
        self._modes: list[str] = []

    @property
    def modes(self) -> list[str] | None:
        """The modes of the text last reported spoken, or None when no text was added."""
        return self._modes if self._text else None

    def add(self, text: str, spans: list[ExpressionSpan]):
        """Keep a unit sent to the service, for the words it reports later.

        Args:
            text: The unit, as the LLM wrote it.
            spans: What each stretch of the unit performs.
        """
        offset = len(self._text)
        self._text += text
        self._spans.extend(
            _Span(offset + span.start, offset + span.end, span.modes, span.event) for span in spans
        )

    def spoken(self, text: str | None) -> tuple[list[str] | None, list[tuple[TextType, str]]]:
        """Read the modes of text the service reports speaking, and the events before it.

        Args:
            text: The LLM's text it covers, tags included.

        Returns:
            Its modes, or None when it isn't text that was added, and the
            events and pauses performed since the text before it, each with
            its text type and its text.
        """
        if not self._text or not text:
            return None, []
        start = self._text.find(text, self._position)
        if start < 0:
            return self._modes, []
        end = start + len(text)
        events = []
        for span in self._spans:
            if span.event and not span.reported and self._position <= span.start < end:
                span.reported = True
                events.append(span.event)
        for i in range(start, end):
            if not self._text[i].isalnum():
                continue
            words = next(
                (s for s in self._spans if s.modes is not None and s.start <= i < s.end),
                None,
            )
            if words is not None:
                self._modes = words.modes or []
                break
        self._position = end
        return self._modes, events

    def spoken_markup(self) -> tuple[TextType, str] | None:
        """Read markup the service reports speaking as a word, such as ``[laugh]``.

        Some services include their markup in their word timestamps. Markup
        stands for the next event or pause still to be spoken, which is
        reported with it, at its time. Markup with no event to stand for, such
        as a mode, isn't spoken.

        Returns:
            The event or pause, with its text type, or None when the markup
            stands for none.
        """
        for span in self._spans:
            if span.end <= self._position or span.reported:
                continue
            if span.event:
                span.reported = True
                return span.event
            words = self._text[max(self._position, span.start) : span.end]
            if any(c.isalnum() for c in words):
                return None
        return None
