#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The expression of the words a TTS service reports speaking.

Part of expressive conversation (alpha). A TTS service that reports word
timestamps tells which words it has spoken, and Pipecat's word tracking gives
each word the stretch of the LLM's text it covers, tags included, as its
``raw_text``. A word covers the text from the end of the word before it, so
tags written between two words arrive with the second.

:class:`SpokenExpression` keeps the text sent in one audio context, with the
:class:`~pipecat.expressive.formatter.ExpressionSpan` list
:meth:`~pipecat.expressive.formatter.ExpressionFormatter.format` returned for
it. Looking up each spoken word's text in it gives the modes the word is
spoken in and the events performed before it, which the TTS service reports
to clients as the word is spoken.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass

from pipecat.expressive.formatter import ExpressionSpan
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
    """Tells the modes and events of the words a TTS service reports speaking in one audio context.

    Each :class:`~pipecat.services.tts_service.TTSContext` holds one, so it
    lasts as long as its audio context. Units are added as they are sent, and
    words are looked up as they are spoken, in order: each lookup searches
    from where the last one ended.

    Example::

        spoken = SpokenExpression()
        spoken.add("<<cheerful>> It's [[laughs]] ^^back^^ up.", spans)

        spoken.spoken("<<cheerful>> It's")
        # (["cheerful"], [])
        spoken.spoken("[[laughs]] ^^back")
        # (["cheerful", "emphasis"], [(TextType.EXPRESSION_EVENT, "laughs")])
        spoken.spoken("up.")
        # (["cheerful"], [])
    """

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

        Units follow one another in the context's text, and the spans are
        moved to where the unit falls in it.

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

        The text's modes are those of its first letter that falls in words,
        so a mode tag's own name, such as ``cheerful`` in
        ``"<<cheerful>> It's"``, isn't taken for a word. An event already
        reported by :meth:`spoken_markup` isn't reported again.

        Args:
            text: The LLM's text it covers, tags included, such as a word's
                ``raw_text``, or a whole sentence's from a service without
                word timestamps.

        Returns:
            Its modes, or None when no text was added, and the events and
            pauses performed since the text before it, each with its text
            type and its text. Text that can't be found keeps the modes of
            the text before it.
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
        """Read markup the service reports speaking as a word, such as ``[laughs]``.

        Some services include their markup in their word timestamps. It
        covers none of the LLM's text, so it has no ``raw_text``, and is
        matched by position instead: it stands for the next event or pause
        still to be spoken, which is reported with it, at its time. Markup
        reached before any event, such as a mode's ``[cheerful]``, stands for
        none, and isn't spoken::

            spoken.spoken("<<cheerful>> It's")
            spoken.spoken_markup()       # "[laughs]" reported as a word
            # (TextType.EXPRESSION_EVENT, "laughs")
            spoken.spoken("[[laughs]] ^^back")
            # (["cheerful", "emphasis"], []): already reported

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
