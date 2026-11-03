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
it. Looking up each spoken word's text in it gives the delivery the word is
spoken in and the sounds performed before it, which the TTS service reports
to clients as the word is spoken.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import replace

from pipecat.expressive.formatter import ExpressionSpan
from pipecat.frames.frames import TextType


class SpokenExpression:
    """Tells the delivery and sounds of the words a TTS service reports speaking in one audio context.

    Each :class:`~pipecat.services.tts_service.TTSContext` holds one, so it
    lasts as long as its audio context. Units are added as they are sent, and
    words are looked up as they are spoken, in order: each lookup searches
    from where the last one ended.

    Example::

        spoken = SpokenExpression()
        spoken.add("<<warm>> It's [[laugh]] ^^back^^ up.", spans)

        spoken.spoken("<<warm>> It's")
        # (["warm"], [])
        spoken.spoken("[[laugh]] ^^back")
        # (["warm", "emphasis"], [(TextType.EXPRESSION_SOUND, "laugh")])
        spoken.spoken("up.")
        # (["warm"], [])
    """

    def __init__(self):
        """Initialize the spoken expression."""
        self._text = ""
        self._spans: list[ExpressionSpan] = []
        # The sounds and pauses already reported, by their index in the spans.
        self._reported: set[int] = set()
        # How far into the text the service has reported speaking.
        self._position = 0
        self._delivery: list[str] = []

    @property
    def delivery(self) -> list[str] | None:
        """The delivery of the text last reported spoken, or None when no text was added."""
        return self._delivery if self._text else None

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
            replace(span, start=offset + span.start, end=offset + span.end) for span in spans
        )

    def spoken(self, text: str | None) -> tuple[list[str] | None, list[tuple[TextType, str]]]:
        """Read the delivery of text the service reports speaking, and the sounds before it.

        The text's delivery is that of its first letter that falls in words,
        so a delivery tag's own name, such as ``warm`` in
        ``"<<warm>> It's"``, isn't taken for a word. A sound already
        reported by :meth:`spoken_markup` isn't reported again, and one
        written after the text's last word, such as the laugh in
        ``"back [[laugh]]!"``, is left for the service's markup or the next
        text to report, since it is performed after the word.

        Args:
            text: The LLM's text it covers, tags included, such as a word's
                ``raw_text``, or a whole sentence's from a service without
                word timestamps.

        Returns:
            Its delivery, or None when no text was added, and the sounds and
            pauses performed since the text before it, each with its text
            type and its text. Text that can't be found keeps the delivery of
            the text before it.
        """
        if not self._text or not text:
            return None, []
        start = self._text.find(text, self._position)
        if start < 0:
            return self._delivery, []
        end = start + len(text)
        last_word = next(
            (i for i in range(end - 1, start - 1, -1) if self._in_words(i)),
            None,
        )
        position = end
        sounds = []
        for i, span in enumerate(self._spans):
            if span.sound and i not in self._reported and self._position <= span.start < end:
                if last_word is not None and span.start > last_word:
                    position = min(position, span.start)
                    continue
                self._reported.add(i)
                sounds.append(span.sound)
        for i in range(start, end):
            if not self._text[i].isalnum():
                continue
            words = next(
                (s for s in self._spans if s.delivery is not None and s.start <= i < s.end),
                None,
            )
            if words is not None:
                self._delivery = words.delivery or []
                break
        self._position = position
        return self._delivery, sounds

    def unreported(self) -> list[tuple[TextType, str]]:
        """Read the sounds and pauses not yet reported, once the context's audio has ended.

        A sound after the last words, such as the laugh in ``"Bye
        [[laugh]]"``, has no later text to report it, and a service may not
        report its markup.

        Returns:
            Each such sound or pause, with its text type and its text.
        """
        sounds = []
        for i, span in enumerate(self._spans):
            if span.sound and i not in self._reported and span.start >= self._position:
                self._reported.add(i)
                sounds.append(span.sound)
        return sounds

    def _in_words(self, i: int) -> bool:
        """Whether the character at ``i`` is a letter or digit of words, not of a tag."""
        return self._text[i].isalnum() and any(
            s.delivery is not None and s.start <= i < s.end for s in self._spans
        )

    def spoken_markup(self) -> tuple[TextType, str] | None:
        """Read markup the service reports speaking as a word, such as ``[laughs]``.

        Some services include their markup in their word timestamps. It
        covers none of the LLM's text, so it has no ``raw_text``, and is
        matched by position instead: it stands for the next sound or pause
        still to be spoken, which is reported with it, at its time, as written in
        its tag. Markup reached before any sound, such as a delivery's
        ``[warmly]``, stands for none, and isn't spoken::

            spoken.spoken("<<warm>> It's")
            spoken.spoken_markup()       # "[laughs]" reported as a word
            # (TextType.EXPRESSION_SOUND, "laugh")
            spoken.spoken("[[laugh]] ^^back")
            # (["warm", "emphasis"], []): already reported

        Returns:
            The sound or pause, with its text type, or None when the markup
            stands for none.
        """
        for i, span in enumerate(self._spans):
            if span.end <= self._position or i in self._reported:
                continue
            if span.sound:
                self._reported.add(i)
                return span.sound
            words = self._text[max(self._position, span.start) : span.end]
            if any(c.isalnum() for c in words):
                return None
        return None
