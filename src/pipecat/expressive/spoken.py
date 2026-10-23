#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The expression of the words a TTS service reports speaking.

Part of expressive conversation (alpha). A TTS service reports the words it
speaks with the stretch of the LLM's text each covers, tags included.
:class:`SpokenExpression` keeps the text sent for each audio context, with
what each stretch of it performs, so each reported word can be given the
modes it is spoken in and the events performed before it.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field

from pipecat.expressive.renderer import ExpressionSpan
from pipecat.frames.frames import TextType


@dataclass
class _Span:
    """What a stretch of the context's text performs: words in their modes, or an event."""

    start: int
    end: int
    modes: list[str] | None = None
    event: tuple[TextType, str] | None = None
    # Whether the event has been reported, as by markup the service reported.
    reported: bool = False


@dataclass
class _SpokenText:
    """The LLM's text sent for one audio context, and how far it has been spoken."""

    text: str = ""
    spans: list[_Span] = field(default_factory=list)
    position: int = 0
    modes: list[str] = field(default_factory=list)


class SpokenExpression:
    """Tells the modes and events of the words a TTS service reports speaking."""

    def __init__(self):
        """Initialize the spoken expression."""
        self._spoken: dict[str, _SpokenText] = {}

    def add(self, context_id: str, text: str, spans: list[ExpressionSpan]):
        """Keep a unit sent to the service, for the words it reports later.

        Args:
            context_id: The audio context the unit is spoken in.
            text: The unit, as the LLM wrote it.
            spans: What each stretch of the unit performs.
        """
        spoken = self._spoken.setdefault(context_id, _SpokenText())
        offset = len(spoken.text)
        spoken.text += text
        spoken.spans.extend(
            _Span(offset + span.start, offset + span.end, span.modes, span.event) for span in spans
        )

    def spoken(
        self, context_id: str | None, text: str | None
    ) -> tuple[list[str] | None, list[tuple[TextType, str]]]:
        """Read the modes of text the service reports speaking, and the events before it.

        Args:
            context_id: The audio context it is spoken in.
            text: The LLM's text it covers, tags included.

        Returns:
            Its modes, or None when it isn't text that was added, and the
            events and pauses performed since the text before it, each with
            its text type and its text.
        """
        spoken = self._spoken.get(context_id) if context_id else None
        if spoken is None or not text:
            return None, []
        start = spoken.text.find(text, spoken.position)
        if start < 0:
            return spoken.modes, []
        end = start + len(text)
        events = []
        for span in spoken.spans:
            if span.event and not span.reported and spoken.position <= span.start < end:
                span.reported = True
                events.append(span.event)
        for i in range(start, end):
            if not spoken.text[i].isalnum():
                continue
            words = next(
                (s for s in spoken.spans if s.modes is not None and s.start <= i < s.end),
                None,
            )
            if words is not None:
                spoken.modes = words.modes or []
                break
        spoken.position = end
        return spoken.modes, events

    def spoken_markup(self, context_id: str | None) -> tuple[TextType, str] | None:
        """Read markup the service reports speaking as a word, such as ``[laugh]``.

        Some services include their markup in their word timestamps. Markup
        stands for the next event or pause still to be spoken, which is
        reported with it, at its time. Markup with no event to stand for, such
        as a mode, isn't spoken.

        Args:
            context_id: The audio context it is spoken in.

        Returns:
            The event or pause, with its text type, or None when the markup
            stands for none.
        """
        spoken = self._spoken.get(context_id) if context_id else None
        if spoken is None:
            return None
        for span in spoken.spans:
            if span.end <= spoken.position or span.reported:
                continue
            if span.event:
                span.reported = True
                return span.event
            words = spoken.text[max(spoken.position, span.start) : span.end]
            if any(c.isalnum() for c in words):
                return None
        return None

    def current_modes(self, context_id: str | None) -> list[str] | None:
        """The modes of the text last reported spoken in an audio context.

        Args:
            context_id: The audio context.

        Returns:
            The modes, or None when the context has no text that was added.
        """
        spoken = self._spoken.get(context_id) if context_id else None
        return spoken.modes if spoken else None

    def remove(self, context_id: str | None):
        """Drop what was sent for an audio context, once it has stopped or been interrupted.

        Args:
            context_id: The audio context.
        """
        if context_id:
            self._spoken.pop(context_id, None)
