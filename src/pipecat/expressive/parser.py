#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Streaming parser for the expression tags an LLM writes into its reply.

Part of expressive conversation (alpha). :class:`ExpressionParser` splits text
into the words to speak and the delivery around them:

- ``[[name]]`` is an :class:`Event`, a sound at one point (``[[laugh]]``), or a
  :class:`Pause`, ``[[pause N]]`` for N milliseconds.
- ``<<name>>`` is a :class:`Mode`, the delivery of the words after it
  (``<<cheerful>>``). ``<<neutral>>`` returns to the voice's baseline.
- ``*words*`` and ``**words**`` are :class:`Emphasis`, following CommonMark's
  rules, so bullets (``* item``) and arithmetic (``3 * 4``) stay text.

The brackets decide a tag's kind, except that a name the TTS service declares
keeps its declared kind in the other kind's brackets: ``[[cheerful]]`` is still
a mode when ``cheerful`` is a declared mode. LLMs make that slip, so it is
logged as a warning. Names are kept as written; the TTS service decides what
it can perform.

Text arrives in pieces as the LLM streams it, so the parser holds back anything
that may be the start of a tag or an emphasis span until it can tell. A tag
left unfinished when the text ends is removed, never spoken.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger

# The mode that returns to the voice's baseline. Part of the protocol, so it is
# a mode whatever a service declares.
NEUTRAL_MODE = "neutral"

# Longest tag or emphasis span, in characters between the delimiters. Text
# that runs longer is not a tag, which bounds how much is held back.
_MAX_SPAN_LENGTH = 64

_PAUSE = re.compile(r"pause (\d+)")
# Characters that end an emphasis span on either side.
_EMPHASIS_BOUNDARY = set(" \t\n.,;:!?\"'()[]{}<>-—–…")


@dataclass(frozen=True)
class Text:
    """Words to speak as they are.

    Parameters:
        text: The words.
    """

    text: str


@dataclass(frozen=True)
class Emphasis:
    """Words to stress.

    Parameters:
        text: The words, without their asterisks.
        raw: The span as written, asterisks included.
    """

    text: str
    raw: str


@dataclass(frozen=True)
class Event:
    """A sound made at one point, such as a laugh.

    Parameters:
        name: The event, as written between the brackets.
        raw: The tag as written.
    """

    name: str
    raw: str


@dataclass(frozen=True)
class Pause:
    """A silence at one point.

    Parameters:
        duration_ms: How long, in milliseconds.
        raw: The tag as written.
    """

    duration_ms: int
    raw: str


@dataclass(frozen=True)
class Mode:
    """The delivery of the words that follow, until the next mode.

    Parameters:
        name: The mode, as written between the brackets.
        raw: The tag as written.
    """

    name: str
    raw: str


Segment = Text | Emphasis | Event | Pause | Mode


class ExpressionParser:
    """Splits streamed LLM text into words and expression tags, in order.

    Feed text as it arrives with :meth:`feed`, call :meth:`flush` when the
    response ends, and :meth:`reset` when it is interrupted. Each returns the
    segments that are complete; a possible tag or emphasis span at the end of
    the text is held back until the text that follows settles it.

    Example::

        parser = ExpressionParser(events=["laugh"], modes=["cheerful"])
        parser.feed("<<cheerful>> It's *back* [[la")
        # [Mode("cheerful", ...), Text("It's "), Emphasis("back", "*back*"), Text(" ")]
        parser.feed("ugh]]!")
        # [Event("laugh", "[[laugh]]"), Text("!")]
    """

    def __init__(self, *, events: Iterable[str] = (), modes: Iterable[str] = ()):
        """Initialize the parser.

        Args:
            events: Names of the events the TTS service declares, so one
                written in mode brackets is still read as an event.
            modes: Names of the modes the TTS service declares, so one written
                in event brackets is still read as a mode.
        """
        self._events = set(events)
        self._modes = {NEUTRAL_MODE, *modes}
        self._buffer = ""
        # The character before the buffer, for emphasis: an opening asterisk
        # needs a space or punctuation before it.
        self._previous_char = ""

    def feed(self, text: str) -> list[Segment]:
        """Add streamed text and return the segments it completes.

        Args:
            text: The next piece of text.

        Returns:
            The complete segments, in order.
        """
        self._buffer += text
        return self._parse(final=False)

    def flush(self) -> list[Segment]:
        """Return the rest of the text at the end of a response.

        A tag left unfinished is removed with a warning; anything else held
        back is returned as text.

        Returns:
            The remaining segments, in order.
        """
        segments = self._parse(final=True)
        self._previous_char = ""
        return segments

    def reset(self):
        """Discard any held-back text, such as a tag cut off by an interruption."""
        self._buffer = ""
        self._previous_char = ""

    def _parse(self, *, final: bool) -> list[Segment]:
        segments: list[Segment] = []
        text: list[str] = []
        buffer = self._buffer
        i = 0
        while i < len(buffer):
            char = buffer[i]
            if char in "[<*":
                result = self._match(buffer, i, final)
                if result is None:
                    break  # Possibly the start of something; wait for more text.
                end, segment = result
                if end > i:
                    if segment is not None:
                        if text:
                            segments.append(Text("".join(text)))
                            text = []
                        segments.append(segment)
                    self._previous_char = " " if not isinstance(segment, Emphasis) else "*"
                    i = end
                    continue
            text.append(char)
            self._previous_char = char
            i += 1
        if text:
            segments.append(Text("".join(text)))
        self._buffer = buffer[i:]
        return segments

    def _match(self, buffer: str, i: int, final: bool) -> tuple[int, Segment | None] | None:
        """Match what starts at ``buffer[i]``.

        Returns:
            ``(end, segment)`` for a tag (``segment`` None when it is
            removed), ``(i, None)`` when it is plain text, or None when the
            text so far can't tell.
        """
        if buffer.startswith("[[", i) or buffer.startswith("<<", i):
            return self._match_tag(buffer, i, final)
        if buffer[i] == "*":
            return self._match_emphasis(buffer, i, final)
        if not final and len(buffer) - i < 2:
            return None  # A single bracket that may become a double one.
        return i, None

    def _match_tag(self, buffer: str, i: int, final: bool) -> tuple[int, Segment | None] | None:
        opener = buffer[i : i + 2]
        closer = "]]" if opener == "[[" else ">>"
        start = i + 2
        end = buffer.find(closer, start, start + _MAX_SPAN_LENGTH + len(closer))
        if end >= 0:
            content = buffer[start:end]
            if _is_tag_content(content):
                raw = buffer[i : end + 2]
                return end + 2, self._classify(content, raw, mode=opener == "<<")
            return i, None
        rest = buffer[start:]
        # The text may end halfway through the closer.
        content = rest[:-1] if not final and rest.endswith(closer[0]) else rest
        if not _is_tag_content(content) or len(content) > _MAX_SPAN_LENGTH:
            return i, None
        if not final:
            return None
        # "a << b" is text; "<<cheer" is a tag the text ended in the middle of.
        if not rest or not rest[0].isspace():
            logger.warning(f"Removing an unfinished expression tag: {buffer[i:]!r}")
            return len(buffer), None
        return i, None

    def _match_emphasis(
        self, buffer: str, i: int, final: bool
    ) -> tuple[int, Segment | None] | None:
        run = 2 if buffer.startswith("**", i) else 1
        if not final and len(buffer) - i <= run:
            return None  # Can't yet tell how long the run is or what follows it.
        after = buffer[i + run] if i + run < len(buffer) else ""
        if (
            after in ("", "*")
            or after.isspace()
            or (self._previous_char and self._previous_char not in _EMPHASIS_BOUNDARY)
        ):
            return i, None
        start = i + run
        delimiter = "*" * run
        j = start
        while True:
            end = buffer.find(delimiter, j, start + _MAX_SPAN_LENGTH + run)
            if end < 0:
                break
            content = buffer[start:end]
            if "*" in content or "\n" in content:
                return i, None
            following = buffer[end + run] if end + run < len(buffer) else ""
            if following == "*":
                return i, None
            if not following and not final:
                return None  # The next character decides whether this closes the span.
            if not buffer[end - 1].isspace() and (not following or following in _EMPHASIS_BOUNDARY):
                return end + run, Emphasis(content, buffer[i : end + run])
            j = end + 1
        # The text may end halfway through the closing asterisks.
        rest = buffer[start:].rstrip("*")
        if not final and "\n" not in rest and "*" not in rest and len(rest) <= _MAX_SPAN_LENGTH:
            return None
        return i, None

    def _classify(self, name: str, raw: str, *, mode: bool) -> Segment:
        if not mode:
            pause = _PAUSE.fullmatch(name)
            if pause:
                return Pause(int(pause.group(1)), raw)
        if mode and name in self._events and name not in self._modes:
            logger.warning(f"Reading an event written in mode brackets as an event: {raw!r}")
            return Event(name, raw)
        if not mode and name in self._modes and name not in self._events:
            logger.warning(f"Reading a mode written in event brackets as a mode: {raw!r}")
            return Mode(name, raw)
        return Mode(name, raw) if mode else Event(name, raw)


def _is_tag_content(text: str) -> bool:
    """Whether text can sit between a tag's delimiters: words, digits and a little punctuation."""
    return all(c.isalnum() or c in " '-:/" for c in text)
