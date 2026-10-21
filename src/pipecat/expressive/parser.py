#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Streaming parser for the expression tags an LLM writes into its reply.

Part of expressive conversation (alpha). :class:`ExpressionParser` splits text
into the words to speak, with the delivery they are to be spoken in, and the
events between them:

- ``<<name ...>>`` sets the modes the words after it are spoken in, until the
  next mode tag or the end of the response: ``<<cheerful>>``, or several
  separated by spaces, ``<<cheerful calm>>``. ``<<neutral>>`` returns to the
  voice's baseline. Mode tags aren't segments of their own; the modes travel
  with the :class:`Text` they apply to.
- ``*words*`` and ``**words**`` stress words, following CommonMark's rules, so
  bullets (``* item``) and arithmetic (``3 * 4``) stay text. Stress is a mode
  too: the words are :class:`Text` with ``emphasis`` added to the modes.
- ``[[name]]`` is an :class:`Event`, a sound at one point (``[[laugh]]``), or a
  :class:`Pause`, ``[[pause N]]`` for N milliseconds.

The brackets decide a tag's kind, except that a name the TTS service declares
keeps its declared kind in the other kind's brackets: ``[[cheerful]]`` still
sets a mode when ``cheerful`` is a declared mode. LLMs make that slip, so it is
logged as a warning. Names are kept as written; the TTS service decides what
it can perform.

Every segment keeps the text it came from in ``raw``, a mode tag included in
the raw text of the segment after it, so the raw text of the segments joins up
to what the LLM wrote.

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

# The mode of stressed words, written *words*.
EMPHASIS_MODE = "emphasis"

# Longest tag or emphasis span, in characters between the delimiters. Text
# that runs longer is not a tag, which bounds how much is held back.
_MAX_SPAN_LENGTH = 64

_PAUSE = re.compile(r"pause (\d+)")
# Characters that end an emphasis span on either side.
_EMPHASIS_BOUNDARY = set(" \t\n.,;:!?\"'()[]{}<>-—–…")


@dataclass(frozen=True)
class Text:
    """Words to speak, and the modes to speak them in.

    Parameters:
        text: The words.
        modes: The modes the words are spoken in, empty for the voice's
            baseline.
        raw: The text as written, tags and asterisks included.
    """

    text: str
    modes: tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class Event:
    """A sound made at one point, such as a laugh.

    Parameters:
        name: The event, as written between the brackets.
        raw: The text as written.
    """

    name: str
    raw: str


@dataclass(frozen=True)
class Pause:
    """A silence at one point.

    Parameters:
        duration_ms: How long, in milliseconds.
        raw: The text as written.
    """

    duration_ms: int
    raw: str


Segment = Text | Event | Pause


@dataclass(frozen=True)
class _ModeTag:
    """A mode tag, which sets the modes rather than becoming a segment."""

    modes: tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class _Emphasis:
    """Stressed words, before the active modes are added to them."""

    text: str
    raw: str


class ExpressionParser:
    """Splits streamed LLM text into words with their modes, and events, in order.

    Feed text as it arrives with :meth:`feed`, call :meth:`flush` when the
    response ends, and :meth:`reset` when it is interrupted. Each returns the
    segments that are complete; a possible tag or emphasis span at the end of
    the text is held back until the text that follows settles it. The modes
    last until the next mode tag, and the end of a response returns them to the
    baseline.

    Example::

        parser = ExpressionParser(events=["laugh"])
        parser.feed("<<cheerful>> It's *back* [[la")
        # [Text(" It's ", ("cheerful",), "<<cheerful>> It's "),
        #  Text("back", ("cheerful", "emphasis"), "*back*"),
        #  Text(" ", ("cheerful",), " ")]
        parser.feed("ugh]]!")
        # [Event("laugh", "[[laugh]]"), Text("!", ("cheerful",), "!")]
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
        # The modes the next words are spoken in.
        self._active_modes: tuple[str, ...] = ()
        # Mode tags read since the last segment, which go into its raw text.
        self._pending_raw = ""

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

        A tag left unfinished is removed with a warning, and a mode tag with no
        words after it is dropped; anything else held back is returned as text.
        The modes return to the baseline.

        Returns:
            The remaining segments, in order.
        """
        segments = self._parse(final=True)
        if self._pending_raw:
            # Nothing follows it to speak in its modes, or to carry it.
            logger.debug(f"Dropping a mode tag that ends the response: {self._pending_raw!r}")
        self.reset()
        return segments

    def reset(self):
        """Discard any held-back text, such as a tag cut off by an interruption."""
        self._buffer = ""
        self._previous_char = ""
        self._active_modes = ()
        self._pending_raw = ""

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
                end, match = result
                if end > i:
                    if text:
                        segments.append(self._text("".join(text)))
                        text = []
                    if isinstance(match, _ModeTag):
                        self._active_modes = match.modes
                        self._pending_raw += match.raw
                    elif isinstance(match, _Emphasis):
                        modes = (*self._active_modes, EMPHASIS_MODE)
                        segments.append(self._text(match.text, modes, match.raw))
                    elif match is not None:
                        segments.append(self._with_pending_raw(match))
                    self._previous_char = "*" if isinstance(match, _Emphasis) else " "
                    i = end
                    continue
            text.append(char)
            self._previous_char = char
            i += 1
        if text:
            segments.append(self._text("".join(text)))
        self._buffer = buffer[i:]
        return segments

    def _text(self, text: str, modes: tuple[str, ...] | None = None, raw: str | None = None):
        """Build a Text segment in the active modes, taking the pending raw text."""
        segment = Text(
            text, self._active_modes if modes is None else modes, raw if raw is not None else text
        )
        return self._with_pending_raw(segment)

    def _with_pending_raw(self, segment: Segment) -> Segment:
        if not self._pending_raw:
            return segment
        pending, self._pending_raw = self._pending_raw, ""
        if isinstance(segment, Text):
            return Text(segment.text, segment.modes, pending + segment.raw)
        if isinstance(segment, Event):
            return Event(segment.name, pending + segment.raw)
        return Pause(segment.duration_ms, pending + segment.raw)

    def _match(
        self, buffer: str, i: int, final: bool
    ) -> tuple[int, Segment | _ModeTag | _Emphasis | None] | None:
        """Match what starts at ``buffer[i]``.

        Returns:
            ``(end, match)`` for a tag or emphasis span (``match`` None when
            it is removed), ``(i, None)`` when it is plain text, or None when
            the text so far can't tell.
        """
        if buffer.startswith("[[", i) or buffer.startswith("<<", i):
            return self._match_tag(buffer, i, final)
        if buffer[i] == "*":
            return self._match_emphasis(buffer, i, final)
        if not final and len(buffer) - i < 2:
            return None  # A single bracket that may become a double one.
        return i, None

    def _match_tag(
        self, buffer: str, i: int, final: bool
    ) -> tuple[int, Segment | _ModeTag | None] | None:
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
    ) -> tuple[int, _Emphasis | None] | None:
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
                return end + run, _Emphasis(content, buffer[i : end + run])
            j = end + 1
        # The text may end halfway through the closing asterisks.
        rest = buffer[start:].rstrip("*")
        if not final and "\n" not in rest and "*" not in rest and len(rest) <= _MAX_SPAN_LENGTH:
            return None
        return i, None

    def _classify(self, content: str, raw: str, *, mode: bool) -> Segment | _ModeTag:
        if not mode:
            pause = _PAUSE.fullmatch(content)
            if pause:
                return Pause(int(pause.group(1)), raw)
        if mode and content in self._events and content not in self._modes:
            logger.warning(f"Reading an event written in mode brackets as an event: {raw!r}")
            return Event(content, raw)
        if not mode and content in self._modes and content not in self._events:
            logger.warning(f"Reading a mode written in event brackets as a mode: {raw!r}")
            return self._mode_tag(content, raw)
        return self._mode_tag(content, raw) if mode else Event(content, raw)

    def _mode_tag(self, content: str, raw: str) -> _ModeTag:
        modes = tuple(content.split())
        return _ModeTag(() if modes == (NEUTRAL_MODE,) else modes, raw)


def _is_tag_content(text: str) -> bool:
    """Whether text can sit between a tag's delimiters: words, digits and a little punctuation."""
    return all(c.isalnum() or c in " '-:/" for c in text)
