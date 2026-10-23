#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reads the expression tags an LLM writes, and formats them for a TTS service.

Part of expressive conversation (alpha). :class:`ExpressionFormatter` splits
each unit of text a TTS service is given into the words to speak, with the
modes they are spoken in, and the events between them:

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
logged as a warning.

The formatter reads whole units: an aggregator such as
:class:`~pipecat.utils.text.skip_tags_aggregator.SkipTagsAggregator`, given
:data:`EXPRESSION_TAGS`, keeps each tag in one unit as text streams. A tag
left unfinished at the end of a unit is removed, never spoken.

It then formats a unit's request through the service's
:class:`~pipecat.adapters.tts.base_expression_adapter.BaseExpressionAdapter`,
leaving out what the service can't perform.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    EMPHASIS_MODE,
    NEUTRAL_MODE,
    ExpressionCapabilities,
    ModeScope,
)
from pipecat.frames.frames import TextType
from pipecat.utils.string import StartEndTags

# The delimiters of the tags, for an aggregator to keep each tag in one unit.
EXPRESSION_TAGS: list[StartEndTags] = [("<<", ">>"), ("[[", "]]"), ("^^", "^^")]

# The tags an LLM writes into its text, which word tracking passes over in it.
EXPRESSION_TAG_PATTERN = re.compile(r"<<[^<>\n]*>>|\[\[[^\[\]\n]*\]\]")

# A tag holds words, digits and a little punctuation; anything else is text.
_TAG = re.compile(r"<<([\w '\-:/]*)>>|\[\[([\w '\-:/]*)\]\]|\^\^([^\n]+?)\^\^")
# What is left of a tag the unit ended in the middle of, and an emphasis that
# never closed.
_UNFINISHED = re.compile(r"(?:<<|\[\[)(?:[\w'\-:/][\w '\-:/]*)?$")
_UNCLOSED_EMPHASIS = "^^"
_PAUSE = re.compile(r"pause (\d+)")

# Spaces and tabs left side by side where a tag stood between two words.
_SPACES = re.compile(r"[ \t]{2,}")


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


@dataclass(frozen=True)
class ExpressionSpan:
    """What a stretch of a unit's text performs: words in their modes, or an event.

    Parameters:
        start: Where the stretch starts in the unit of text.
        end: Where it ends.
        modes: The modes its words are spoken in, or None for an event.
        event: The event or pause, with its text type, as reported to
            clients, or None for words.
    """

    start: int
    end: int
    modes: list[str] | None = None
    event: tuple[TextType, str] | None = None


class ExpressionFormatter:
    """Reads a TTS service's expression tags and formats its requests.

    Parse each unit of a response in turn with :meth:`parse`, and call
    :meth:`reset` when the response ends or is interrupted. The modes last
    until the next mode tag, and the end of a response returns them to the
    baseline. :meth:`format` builds a unit's request: the words, transformed
    and formatted in the modes the service can perform, and its events and
    pauses. What the service can't perform is left out, with one warning per
    tag.

    Example::

        formatter = ExpressionFormatter(lambda: adapter, owner=tts)
        formatter.parse("<<cheerful>> It's ^^back^^ [[laugh]]!")
        # [Text(" It's ", ("cheerful",), 12, 18),
        #  Text("back", ("cheerful", "emphasis"), 20, 24),
        #  Text(" ", ("cheerful",), 26, 27),
        #  Event("laugh", 27, 36),
        #  Text("!", ("cheerful",), 36, 37)]
    """

    def __init__(self, adapter: Callable[[], BaseExpressionAdapter], owner: object):
        """Initialize the formatter.

        Args:
            adapter: Returns the service's adapter, which may depend on its
                settings, such as its model.
            owner: The service, named in warnings.
        """
        self._adapter = adapter
        self._owner = owner
        # The modes the next words of the response are spoken in.
        self._active_modes: tuple[str, ...] = ()
        # The modes of the words last formatted, so markup marks only changes.
        self._formatted_modes: list[str] | None = None
        self._unperformed: set[str] = set()
        self._modes: list[str] = []

    @property
    def modes(self) -> list[str]:
        """The modes the latest request is spoken in, empty for the voice's baseline."""
        return self._modes

    def capabilities(self) -> ExpressionCapabilities | None:
        """The expression the service can perform now.

        Returns:
            The service's adapter's capabilities, or None when it performs none.
        """
        return self._adapter().capabilities()

    def parse(self, text: str, *, whole: bool = False) -> list[Segment]:
        """Split a unit of text into segments.

        Args:
            text: The unit, with each of its tags whole.
            whole: Whether the text stands on its own, apart from any
                response, such as a ``TTSSpeakFrame``'s. It starts in the
                baseline and leaves the response's modes as they were.

        Returns:
            Its segments, in order. Spacing alone between two tags is a
            :class:`Text` of its own.
        """
        response_modes = self._active_modes
        if whole:
            self._active_modes = ()
        # The names the service declares keep their kind in either brackets.
        capabilities = self.capabilities()
        declared = capabilities or ExpressionCapabilities()
        events = {term.name for term in declared.events}
        modes = {NEUTRAL_MODE, *(term.name for term in declared.modes)}
        segments: list[Segment] = []
        position = 0
        for match in _TAG.finditer(text):
            self._add_text(segments, text, position, match.start())
            mode_tag, event_tag, stressed = match.groups()
            if stressed is not None:
                stressed_modes = (*self._active_modes, EMPHASIS_MODE)
                segments.append(Text(stressed, stressed_modes, match.start(3), match.end(3)))
            else:
                self._add_tag(segments, mode_tag, event_tag, match, events, modes)
            position = match.end()
        self._add_text(segments, text, position, len(text))
        if whole:
            self._active_modes = response_modes
        return segments

    def restate_modes(self):
        """Have the next request state its modes from its start.

        For text spoken in a context of its own, such as a ``TTSSpeakFrame``'s.
        """
        self._formatted_modes = None

    def reset(self):
        """Return to the baseline, as at the end of a response or an interruption."""
        self._active_modes = ()
        self._formatted_modes = None

    @staticmethod
    def shown_text(segments: list[Segment], text_type: TextType | str) -> str:
        """The words of a unit, without its tags, spaced as written.

        Args:
            segments: The unit's segments.
            text_type: The unit's text type. A sentence's words are trimmed at
                both ends; a streamed token keeps its spacing.

        Returns:
            The words, with one space where a tag stood between two.
        """
        words = _SPACES.sub(" ", "".join(s.text for s in segments if isinstance(s, Text)))
        return words if text_type == TextType.TOKEN else words.strip()

    async def format(
        self,
        segments: list[Segment],
        text_type: TextType | str,
        transform: Callable[[str, TextType | str], Awaitable[str | None]],
        trim: Callable[[str], str | None],
    ) -> tuple[str, list[ExpressionSpan]]:
        """Build the request for a unit.

        Each run of words is transformed, then formatted in the modes the
        service can perform. Events and pauses are formatted at their places.
        A sentence's request is trimmed at both ends; a streamed token keeps
        its spacing.

        Args:
            segments: The unit's segments.
            text_type: The unit's text type.
            transform: Rewrites a run of words for the service, or returns
                None when nothing is left of it.
            trim: Trims the unit's first words as the service trims a unit's
                text, or returns None when nothing is left of them.

        Returns:
            The text to send to the service, empty when nothing is left to
            say, and what each stretch of the unit performs.
        """
        capabilities = self.capabilities()
        if text_type != TextType.TOKEN:
            # A sentence's request states its modes from its start, since not
            # every service keeps modes from one request to the next.
            self._formatted_modes = None
        parts: list[str] = []
        spans: list[ExpressionSpan] = []
        first_modes: list[str] | None = None
        trimmed = False
        for segment in segments:
            if isinstance(segment, Text):
                modes = self._performed_modes(list(segment.modes), capabilities)
                if first_modes is None and segment.text.strip():
                    first_modes = modes
                if capabilities and capabilities.mode_scope == ModeScope.SENTENCE:
                    # One request takes one set of modes: the ones its words
                    # start in. Clients are told the modes that were requested.
                    modes = first_modes if first_modes is not None else modes
                spans.append(ExpressionSpan(segment.start, segment.end, modes=modes))
                if not segment.text.strip():
                    parts.append(segment.text)
                    continue
                words_text: str | None = segment.text
                if not trimmed:
                    # The unit's first words are trimmed as its text would be, and
                    # a sentence's even after an event or pause written before them.
                    first = segment.text if text_type == TextType.TOKEN else segment.text.lstrip()
                    words_text = trim(first)
                    trimmed = True
                words = await transform(words_text, text_type) if words_text is not None else None
                if words is not None:
                    parts.append(self._format(words, modes))
            elif _performs(segment, capabilities):
                event = _event_text(segment, capabilities)
                spans.append(ExpressionSpan(segment.start, segment.end, event=event))
                parts.append(self._format_event(segment, capabilities))
            else:
                self._warn_unperformed(_tag(segment))
        if first_modes is not None:
            self._modes = first_modes
        request = _SPACES.sub(" ", "".join(parts))
        return (request if text_type == TextType.TOKEN else request.strip()), spans

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
        events: set[str],
        modes: set[str],
    ):
        content = (mode_tag if mode_tag is not None else event_tag or "").strip()
        if mode_tag is None:
            pause = _PAUSE.fullmatch(content)
            if pause:
                segments.append(Pause(int(pause.group(1)), match.start(), match.end()))
                return
            if content in modes and content not in events:
                logger.warning(f"Reading a mode written in event brackets as a mode: {match[0]!r}")
                self._set_modes(content)
                return
            segments.append(Event(content, match.start(), match.end()))
            return
        if content in events and content not in modes:
            logger.warning(f"Reading an event written in mode brackets as an event: {match[0]!r}")
            segments.append(Event(content, match.start(), match.end()))
            return
        self._set_modes(content)

    def _set_modes(self, content: str):
        modes = tuple(content.split())
        self._active_modes = () if modes == (NEUTRAL_MODE,) else modes

    def _format(self, text: str, modes: list[str]) -> str:
        """Format words to send to the TTS in their modes."""
        # Leading spacing stays outside the markup, between the words.
        words = text.lstrip()
        formatted = self._adapter().format_text(words, modes, self._formatted_modes)
        self._formatted_modes = modes
        return text[: len(text) - len(words)] + formatted

    def _performed_modes(
        self, modes: list[str], capabilities: ExpressionCapabilities | None
    ) -> list[str]:
        """The modes the service can perform, warning once about each it can't."""
        performed = []
        for mode in modes:
            if mode == EMPHASIS_MODE:
                supported = bool(capabilities and capabilities.emphasis)
            else:
                supported = bool(
                    capabilities
                    and (
                        capabilities.free_text_modes
                        or any(term.name == mode for term in capabilities.modes)
                    )
                )
            if supported:
                performed.append(mode)
            else:
                self._warn_unperformed("^^emphasis^^" if mode == EMPHASIS_MODE else f"<<{mode}>>")
        return performed

    def _format_event(
        self, segment: Event | Pause, capabilities: ExpressionCapabilities | None
    ) -> str:
        """Format an event or pause the service can perform."""
        if isinstance(segment, Pause):
            assert capabilities
            return self._adapter().format_pause(_pause_ms(segment, capabilities))
        return self._adapter().format_event(segment.name)

    def _warn_unperformed(self, tag: str):
        if tag not in self._unperformed:
            self._unperformed.add(tag)
            logger.warning(f"{self._owner} can't perform {tag}, leaving it out of its speech")


def _performs(segment: Event | Pause, capabilities: ExpressionCapabilities | None) -> bool:
    """Whether the service can perform an event or pause."""
    if not capabilities:
        return False
    if isinstance(segment, Pause):
        return capabilities.pause_range_ms is not None
    return capabilities.free_text_events or any(
        term.name == segment.name for term in capabilities.events
    )


def _pause_ms(pause: Pause, capabilities: ExpressionCapabilities) -> int:
    """A pause's length, within the service's range."""
    assert capabilities.pause_range_ms
    shortest, longest = capabilities.pause_range_ms
    return min(max(pause.duration_ms, shortest), longest)


def _event_text(
    segment: Event | Pause, capabilities: ExpressionCapabilities | None
) -> tuple[TextType, str]:
    """An event or pause as reported to clients, with its text type."""
    if isinstance(segment, Pause):
        assert capabilities
        return TextType.EXPRESSION_PAUSE, str(_pause_ms(segment, capabilities))
    return TextType.EXPRESSION_EVENT, segment.name


def _tag(segment: Event | Pause) -> str:
    """An event or pause written as a tag."""
    if isinstance(segment, Pause):
        return f"[[pause {segment.duration_ms}]]"
    return f"[[{segment.name}]]"
