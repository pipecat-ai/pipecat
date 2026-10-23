#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parser for the expression tags an LLM writes into its reply.

Part of expressive conversation (alpha). :class:`ExpressionParser` splits a
unit of text into the words to speak, with the delivery they are to be
spoken in, and the events between them:

- ``<<name ...>>`` sets the modes the words after it are spoken in, until the
  next mode tag or the end of the response: ``<<cheerful>>``, or several
  separated by spaces, ``<<cheerful calm>>``. ``<<neutral>>`` returns to the
  voice's baseline. Mode tags aren't segments of their own; the modes travel
  with the :class:`Text` they apply to.
- ``^^words^^`` stresses words. Stress is a mode too: the words are
  :class:`Text` with ``emphasis`` added to the modes.
- ``[[name]]`` is an :class:`Event`, a sound at one point (``[[laugh]]``), or a
  :class:`Pause`, ``[[pause N]]`` for N milliseconds.

The brackets decide a tag's kind, except that a name the TTS service declares
keeps its declared kind in the other kind's brackets: ``[[cheerful]]`` still
sets a mode when ``cheerful`` is a declared mode. LLMs make that slip, so it is
logged as a warning. Names are kept as written; the TTS service decides what
it can perform.

The parser reads whole units: an aggregator such as
:class:`~pipecat.utils.text.skip_tags_aggregator.SkipTagsAggregator`, given
:data:`EXPRESSION_TAGS`, keeps each tag in one unit as text streams. A tag
left unfinished at the end of a unit is removed, never spoken.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger

from pipecat.utils.string import StartEndTags

# The mode that returns to the voice's baseline. Part of the protocol, so it is
# a mode whatever a service declares.
NEUTRAL_MODE = "neutral"

# The mode of stressed words, written ^^words^^.
EMPHASIS_MODE = "emphasis"

# The delimiters of the tags, for an aggregator to keep each tag in one unit.
EXPRESSION_TAGS: list[StartEndTags] = [("<<", ">>"), ("[[", "]]"), ("^^", "^^")]

# A tag holds words, digits and a little punctuation; anything else is text.
_TAG = re.compile(r"<<([\w '\-:/]*)>>|\[\[([\w '\-:/]*)\]\]|\^\^([^\n]+?)\^\^")
# What is left of a tag the unit ended in the middle of, and an emphasis that
# never closed.
_UNFINISHED = re.compile(r"(?:<<|\[\[)(?:[\w'\-:/][\w '\-:/]*)?$")
_UNCLOSED_EMPHASIS = "^^"
_PAUSE = re.compile(r"pause (\d+)")


@dataclass(frozen=True)
class Text:
    """Words to speak, and the modes to speak them in.

    Parameters:
        text: The words.
        modes: The modes the words are spoken in, empty for the voice's
            baseline.
        start: Where the words start in the unit of text.
        end: Where they end.
    """

    text: str
    modes: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Event:
    """A sound made at one point, such as a laugh.

    Parameters:
        name: The event, as written between the brackets.
        start: Where its tag starts in the unit of text.
        end: Where it ends.
    """

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Pause:
    """A silence at one point.

    Parameters:
        duration_ms: How long, in milliseconds.
        start: Where its tag starts in the unit of text.
        end: Where it ends.
    """

    duration_ms: int
    start: int
    end: int


Segment = Text | Event | Pause


class ExpressionParser:
    """Splits units of LLM text into words with their modes, and events, in order.

    Parse each unit of a response in turn with :meth:`parse`, and call
    :meth:`reset` when the response ends or is interrupted. The modes last
    until the next mode tag, and the end of a response returns them to the
    baseline.

    Example::

        parser = ExpressionParser(events=["laugh"])
        parser.parse("<<cheerful>> It's ^^back^^ [[laugh]]!")
        # [Text(" It's ", ("cheerful",), 12, 18),
        #  Text("back", ("cheerful", "emphasis"), 20, 24),
        #  Text(" ", ("cheerful",), 26, 27),
        #  Event("laugh", 27, 36),
        #  Text("!", ("cheerful",), 36, 37)]
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
        # The modes the next words are spoken in.
        self._active_modes: tuple[str, ...] = ()

    def parse(self, text: str) -> list[Segment]:
        """Split a unit of text into segments.

        Args:
            text: The unit, with each of its tags whole.

        Returns:
            Its segments, in order. Spacing alone between two tags is a
            :class:`Text` of its own.
        """
        segments: list[Segment] = []
        position = 0
        for match in _TAG.finditer(text):
            self._add_text(segments, text, position, match.start())
            mode_tag, event_tag, stressed = match.groups()
            if stressed is not None:
                modes = (*self._active_modes, EMPHASIS_MODE)
                segments.append(Text(stressed, modes, match.start(3), match.end(3)))
            else:
                self._add_tag(segments, mode_tag, event_tag, match)
            position = match.end()
        self._add_text(segments, text, position, len(text))
        return segments

    def reset(self):
        """Return to the baseline, as at the end of a response."""
        self._active_modes = ()

    def _add_text(self, segments: list[Segment], text: str, start: int, end: int):
        words = text[start:end]
        unfinished = _UNFINISHED.search(words) if end == len(text) else None
        if unfinished:
            logger.warning(
                f"Removing an unfinished expression tag: {words[unfinished.start() :]!r}"
            )
            words = words[: unfinished.start()]
            end = start + len(words)
        words = words.replace(_UNCLOSED_EMPHASIS, "")
        if words:
            segments.append(Text(words, self._active_modes, start, end))

    def _add_tag(
        self,
        segments: list[Segment],
        mode_tag: str | None,
        event_tag: str | None,
        match: re.Match[str],
    ):
        content = (mode_tag if mode_tag is not None else event_tag or "").strip()
        if mode_tag is None:
            pause = _PAUSE.fullmatch(content)
            if pause:
                segments.append(Pause(int(pause.group(1)), match.start(), match.end()))
                return
            if content in self._modes and content not in self._events:
                logger.warning(f"Reading a mode written in event brackets as a mode: {match[0]!r}")
                self._set_modes(content)
                return
            segments.append(Event(content, match.start(), match.end()))
            return
        if content in self._events and content not in self._modes:
            logger.warning(f"Reading an event written in mode brackets as an event: {match[0]!r}")
            segments.append(Event(content, match.start(), match.end()))
            return
        self._set_modes(content)

    def _set_modes(self, content: str):
        modes = tuple(content.split())
        self._active_modes = () if modes == (NEUTRAL_MODE,) else modes
