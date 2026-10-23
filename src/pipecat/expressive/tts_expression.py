#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a TTS service performs the expression tags in the text it speaks.

Part of expressive conversation (alpha). A :class:`TTSExpression` reads the
tags in the text a TTS service is given, decides what the service can
perform, and renders it through the service's
:class:`~pipecat.adapters.tts.base_expression_adapter.BaseExpressionAdapter`.
Each unit of text stays one frame: its words are shown, the text as the LLM
wrote it is kept for the context, and the request carries the service's
markup. As the service reports the words it speaks, the expression reports
the modes each is spoken in and the events performed before it.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ModeScope
from pipecat.expressive.parser import EMPHASIS_MODE, Event, ExpressionParser, Pause, Segment, Text
from pipecat.frames.frames import TextType

# The tags an LLM writes into its text, which word tracking passes over in it.
EXPRESSION_TAG_PATTERN = re.compile(r"<<[^<>\n]*>>|\[\[[^\[\]\n]*\]\]")

# Spaces and tabs left side by side where a tag stood between two words.
_SPACES = re.compile(r"[ \t]{2,}")


@dataclass
class _Span:
    """What a stretch of the LLM's text performs: words in their modes, or an event."""

    start: int
    end: int
    # Where the words start, past any mode tags written before them.
    words_start: int = 0
    modes: list[str] | None = None
    event: tuple[TextType, str] | None = None
    # Whether the event has been reported, as by the markup the service echoed.
    reported: bool = False


@dataclass
class _SpokenText:
    """The LLM's text sent for one audio context, and how far it has been spoken."""

    text: str = ""
    spans: list[_Span] = field(default_factory=list)
    position: int = 0
    modes: list[str] = field(default_factory=list)


class TTSExpression:
    """A TTS service's expression: reads, filters and renders its expression tags.

    It reads a response's tags across the units the service is given, as they
    stream, and builds each unit's request: the words, transformed and
    rendered in the modes the service can perform, and its events and pauses.
    What the service can't perform is left out, with one warning per tag. It
    keeps the text it sent for each audio context, so the words the service
    reports can be given the modes they were spoken in.
    """

    def __init__(self, adapter: Callable[[], BaseExpressionAdapter], owner: object):
        """Initialize the service's expression.

        Args:
            adapter: Returns the service's adapter, which may depend on its
                settings, such as its model.
            owner: The service, named in warnings.
        """
        self._adapter = adapter
        self._owner = owner
        # Built on first use, so a service can learn its capabilities at start.
        self._parser: ExpressionParser | None = None
        # The modes of the words last rendered, so markup marks only changes.
        self._rendered_modes: list[str] | None = None
        self._unperformed: set[str] = set()
        self._modes: list[str] = []
        self._spoken: dict[str, _SpokenText] = {}

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

    def feed(self, text: str) -> list[Segment]:
        """Read text of the response being spoken.

        Args:
            text: The next unit of the response.

        Returns:
            The segments complete so far.
        """
        return self._response_parser().feed(text)

    def flush(self) -> list[Segment]:
        """Return what remains of the response, at its end.

        Returns:
            The segments left in the response.
        """
        return self._response_parser().flush()

    def read_whole(self, text: str) -> list[Segment]:
        """Read text that stands on its own, apart from any response.

        Args:
            text: The text, such as a ``TTSSpeakFrame``'s.

        Returns:
            Its segments.
        """
        parser = self._new_parser()
        return parser.feed(text) + parser.flush()

    def restate_modes(self):
        """Have the next request state its modes from its start.

        For the request after a response ends, and for text spoken in a
        context of its own, such as a ``TTSSpeakFrame``'s.
        """
        self._rendered_modes = None

    def reset(self):
        """Drop what remains of the response and what was sent, as on an interruption."""
        if self._parser:
            self._parser.reset()
        self._rendered_modes = None
        self._spoken.clear()

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

    @staticmethod
    def written_text(segments: list[Segment], text_type: TextType | str) -> str:
        """A unit's text as the LLM wrote it, tags included, for the context.

        Args:
            segments: The unit's segments.
            text_type: The unit's text type. A sentence's text is trimmed at
                both ends; a streamed token keeps its spacing.

        Returns:
            The text.
        """
        text = "".join(s.raw for s in segments)
        return text if text_type == TextType.TOKEN else text.strip()

    async def request(
        self,
        segments: list[Segment],
        context_id: str,
        text_type: TextType | str,
        transform: Callable[[str, TextType | str], Awaitable[str | None]],
        trim: Callable[[str], str | None],
    ) -> str:
        """Build the request for a unit, and keep its text for the words reported later.

        Each run of words is transformed, then rendered in the modes the
        service can perform. Events and pauses are rendered at their places.
        A sentence's request is trimmed at both ends; a streamed token keeps
        its spacing.

        Args:
            segments: The unit's segments.
            context_id: The audio context the request is spoken in.
            text_type: The unit's text type.
            transform: Rewrites a run of words for the service, or returns
                None when nothing is left of it.
            trim: Trims the unit's first words as the service trims a unit's
                text, or returns None when nothing is left of them.

        Returns:
            The text to send to the service, empty when nothing is left to say.
        """
        capabilities = self.capabilities()
        if text_type != TextType.TOKEN:
            # A sentence's request states its modes from its start, since not
            # every service keeps modes from one request to the next.
            self._rendered_modes = None
        spoken = self._spoken.setdefault(context_id, _SpokenText())
        parts: list[str] = []
        first_modes: list[str] | None = None
        trimmed = False
        for segment in segments:
            start = len(spoken.text)
            spoken.text += segment.raw
            span = _Span(start, len(spoken.text))
            if isinstance(segment, Text):
                modes = self._performed_modes(list(segment.modes), capabilities)
                if first_modes is None and segment.text.strip():
                    first_modes = modes
                if capabilities and capabilities.mode_scope == ModeScope.SENTENCE:
                    # One request takes one set of modes: the ones its words
                    # start in. Clients are told the modes that were requested.
                    modes = first_modes if first_modes is not None else modes
                span.words_start = start + max(segment.raw.rfind(segment.text.strip()), 0)
                span.modes = modes
                spoken.spans.append(span)
                if not segment.text.strip():
                    parts.append(segment.text)
                    continue
                text = segment.text
                if not trimmed:
                    # The unit's first words are trimmed as its text would be, and
                    # a sentence's even after an event or pause written before them.
                    text = trim(text if text_type == TextType.TOKEN else text.lstrip())
                    trimmed = True
                words = await transform(text, text_type) if text is not None else None
                if words is not None:
                    parts.append(self._render(words, modes))
            elif _performs(segment, capabilities):
                span.event = _event_text(segment, capabilities)
                spoken.spans.append(span)
                parts.append(self._render_event(segment, capabilities))
            else:
                self._warn_unperformed(segment.raw)
        if first_modes is not None:
            self._modes = first_modes
        request = _SPACES.sub(" ", "".join(parts))
        return request if text_type == TextType.TOKEN else request.strip()

    def spoken(
        self, context_id: str | None, text: str | None
    ) -> tuple[list[str] | None, list[tuple[TextType, str]]]:
        """Read the modes of text the service reports speaking, and the events before it.

        Args:
            context_id: The audio context it is spoken in.
            text: The LLM's text it covers, tags included.

        Returns:
            Its modes, or None when it isn't text this expression sent, and
            the events and pauses performed since the text before it, each
            with its text type and its text.
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
                (s for s in spoken.spans if s.modes is not None and s.words_start <= i < s.end),
                None,
            )
            if words is not None:
                spoken.modes = words.modes or []
                break
        spoken.position = end
        return spoken.modes, events

    def echoed(self, context_id: str | None) -> tuple[TextType, str] | None:
        """Read markup the service echoed as a word, such as ``[laugh]``.

        Markup stands for the next event or pause still to be spoken, which is
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
            words = spoken.text[max(spoken.position, span.words_start) : span.end]
            if any(c.isalnum() for c in words):
                return None
        return None

    def current_modes(self, context_id: str | None) -> list[str] | None:
        """The modes of the text last reported spoken in an audio context.

        Args:
            context_id: The audio context.

        Returns:
            The modes, or None when the context has no text this expression sent.
        """
        spoken = self._spoken.get(context_id) if context_id else None
        return spoken.modes if spoken else None

    def forget(self, context_id: str | None):
        """Drop what was sent for an audio context, once it has stopped.

        Args:
            context_id: The audio context.
        """
        if context_id:
            self._spoken.pop(context_id, None)

    def _render(self, text: str, modes: list[str]) -> str:
        """Render words to send to the TTS in their modes."""
        # Leading spacing stays outside the markup, between the words.
        words = text.lstrip()
        rendered = self._adapter().render_text(words, modes, self._rendered_modes)
        self._rendered_modes = modes
        return text[: len(text) - len(words)] + rendered

    def _response_parser(self) -> ExpressionParser:
        if self._parser is None:
            self._parser = self._new_parser()
        return self._parser

    def _new_parser(self) -> ExpressionParser:
        """Create a parser that knows the names the service declares."""
        capabilities = self.capabilities()
        if not capabilities:
            return ExpressionParser()
        return ExpressionParser(
            events=[term.name for term in capabilities.events],
            modes=[term.name for term in capabilities.modes],
        )

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
                self._warn_unperformed("^emphasis^" if mode == EMPHASIS_MODE else f"<<{mode}>>")
        return performed

    def _render_event(
        self, segment: Event | Pause, capabilities: ExpressionCapabilities | None
    ) -> str:
        """Render an event or pause the service can perform."""
        if isinstance(segment, Pause):
            assert capabilities
            return self._adapter().render_pause(_pause_ms(segment, capabilities))
        return self._adapter().render_event(segment.name)

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
