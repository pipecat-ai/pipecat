#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a TTS service performs the expression tags in the text it speaks.

Part of expressive conversation (alpha). A :class:`TTSExpression` reads the
tags in the text a TTS service is given, decides what the service can
perform, and renders it through the service's
:class:`~pipecat.expressive.adapter.ExpressionAdapter`. The TTS service speaks
the pieces it returns, one request per unit of text.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import unicodedata
from collections.abc import Callable

from loguru import logger

from pipecat.expressive.adapter import ExpressionAdapter
from pipecat.expressive.expression import EXPRESSION_TEXT_TYPES, ExpressionCapabilities, ModeScope
from pipecat.expressive.parser import EMPHASIS_MODE, Event, ExpressionParser, Pause, Segment, Text
from pipecat.frames.frames import AggregatedTextFrame, TextType


class TTSExpression:
    """A TTS service's expression: reads, filters and renders its expression tags.

    It reads a response's tags across the units the service is given, as they
    stream, and turns each unit into pieces: its words, in the modes the
    service can perform, and its events and pauses, rendered. What the service
    can't perform is left out, with one warning per tag, so the request and
    what clients see agree.
    """

    def __init__(self, adapter: Callable[[], ExpressionAdapter], owner: object):
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

    def restate_modes(self):
        """Have the next request state its modes from its start.

        For the request after a response ends, and for text spoken in a
        context of its own, such as a ``TTSSpeakFrame``'s.
        """
        self._rendered_modes = None

    def reset(self):
        """Drop what remains of the response, as on an interruption."""
        if self._parser:
            self._parser.reset()
        self._rendered_modes = None

    def read_whole(self, text: str) -> list[Segment]:
        """Read text that stands on its own, apart from any response.

        Args:
            text: The text, such as a ``TTSSpeakFrame``'s.

        Returns:
            Its segments.
        """
        parser = self._new_parser()
        return parser.feed(text) + parser.flush()

    def pieces(
        self,
        src_frame: AggregatedTextFrame,
        segments: list[Segment],
        *,
        streaming_tokens: bool,
        trim: Callable[[str], str | None],
    ) -> list[tuple[AggregatedTextFrame, str]]:
        """Turn a unit's segments into the pieces to speak as one request.

        Its words, in their modes, and its events become pieces of their own,
        in the order written. Unsupported modes are taken off the words, and
        an unsupported event is dropped.

        Args:
            src_frame: The unit the segments come from.
            segments: The unit's segments.
            streaming_tokens: Whether the unit is a streamed token rather than
                a sentence.
            trim: Trims the unit's first word piece, as the unit's own text
                would be trimmed, or returns None when nothing is left to say.

        Returns:
            Each piece's frame and the text to send to the TTS for it.
        """
        capabilities = self.capabilities()
        if not streaming_tokens:
            # A sentence's request states its modes from its start, since not
            # every service keeps modes from one request to the next.
            self._rendered_modes = None
        pieces: list[tuple[AggregatedTextFrame, str]] = []
        trimmed_start = False
        for frame, text in _segment_frames(
            segments, src_frame.text_type, trim=not streaming_tokens
        ):
            frame.interruptible = src_frame.interruptible
            if frame.text_type in EXPRESSION_TEXT_TYPES:
                rendered = self._render_event(frame, capabilities)
                if rendered is not None:
                    pieces.append((frame, rendered))
                continue
            if not trimmed_start:
                # The unit is trimmed at its start; between its pieces, spacing
                # is kept as written.
                trimmed = trim(text)
                if trimmed is None:
                    continue
                text = trimmed
                trimmed_start = True
            elif pieces and pieces[-1][1][-1:].isspace() and text[:1].isspace():
                # An event left out between two spaces leaves one.
                text = text.lstrip()
            frame.modes = self._performed_modes(frame.modes or [], capabilities)
            pieces.append((frame, text))
        words = [frame for frame, _ in pieces if frame.text_type not in EXPRESSION_TEXT_TYPES]
        if words:
            self._modes = words[0].modes or []
            if capabilities and capabilities.mode_scope == ModeScope.SENTENCE:
                # One request takes one set of modes: the ones its words start
                # in. Clients see the modes that were requested.
                for frame in words[1:]:
                    frame.modes = list(self._modes)
        return pieces

    def render(self, text: str, modes: list[str]) -> str:
        """Render words to send to the TTS in their modes.

        Args:
            text: The words, as prepared for the service.
            modes: The modes they are spoken in.

        Returns:
            The text with the service's markup.
        """
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
        self, frame: AggregatedTextFrame, capabilities: ExpressionCapabilities | None
    ) -> str | None:
        """Render an event or pause frame, or None if the service can't perform it."""
        if frame.text_type == TextType.EXPRESSION_PAUSE:
            if capabilities and capabilities.pause_range_ms:
                shortest, longest = capabilities.pause_range_ms
                duration_ms = min(max(int(frame.text), shortest), longest)
                return self._adapter().render_pause(duration_ms)
        elif capabilities and (
            capabilities.free_text_events
            or any(term.name == frame.text for term in capabilities.events)
        ):
            return self._adapter().render_event(frame.text)
        self._warn_unperformed(frame.raw_text or frame.text)
        return None

    def _warn_unperformed(self, tag: str):
        if tag not in self._unperformed:
            self._unperformed.add(tag)
            logger.warning(f"{self._owner} can't perform {tag}, leaving it out of its speech")


def _segment_frames(
    segments: list[Segment], text_type: TextType | str, *, trim: bool
) -> list[tuple[AggregatedTextFrame, str]]:
    """Build the frame for each segment, with the text to send to the TTS for it.

    Words become frames of ``text_type`` that carry their modes, and the text
    to speak keeps the spaces between them. Spacing alone, between two pieces,
    goes with the piece after it, and punctuation written right after a word
    piece goes with that word, so the context joins the pieces as written.
    Events and pauses become frames of their own text types, with no text to
    send until the TTS service renders them.

    Args:
        segments: The segments of one unit of text, such as a sentence.
        text_type: The text type of the unit's words.
        trim: Whether to trim the unit as a sentence is trimmed: the frames'
            text, and the text to speak at the ends of the unit.

    Returns:
        Each segment's frame, with the text to send to the TTS for it.
    """
    pieces: list[tuple[AggregatedTextFrame, str]] = []
    # Spacing alone, with the raw text around it, waiting for the next word piece.
    carried_text = carried_raw = ""
    carried_modes: tuple[str, ...] = ()
    for segment in segments:
        if not isinstance(segment, Text):
            pieces.append((_event_frame(segment), ""))
            continue
        text, raw = segment.text, segment.raw
        previous = pieces[-1][0] if pieces else None
        punctuation = _leading_punctuation(text)
        if (
            punctuation
            and raw.startswith(punctuation)
            and previous is not None
            and previous.text_type not in EXPRESSION_TEXT_TYPES
        ):
            previous.text += punctuation
            previous.raw_text = (previous.raw_text or "") + punctuation
            pieces[-1] = (previous, pieces[-1][1] + punctuation)
            text, raw = text[len(punctuation) :], raw[len(punctuation) :]
        text, raw = carried_text + text, carried_raw + raw
        if not text.strip():
            carried_text, carried_raw, carried_modes = text, raw, segment.modes
            continue
        carried_text = carried_raw = ""
        frame_text, frame_raw = (text.strip(), raw.strip()) if trim else (text, raw)
        frame = AggregatedTextFrame(
            frame_text, text_type, raw_text=frame_raw, modes=list(segment.modes)
        )
        pieces.append((frame, text))
    if carried_text:
        last = pieces[-1][0] if pieces else None
        if last is not None and last.text_type not in EXPRESSION_TEXT_TYPES:
            pieces[-1] = (last, pieces[-1][1] + carried_text)
        elif not trim:
            # A streamed token of spacing alone is sent as it is.
            frame = AggregatedTextFrame(
                carried_text, text_type, raw_text=carried_raw, modes=list(carried_modes)
            )
            pieces.append((frame, carried_text))

    words = [
        i for i, (frame, _) in enumerate(pieces) if frame.text_type not in EXPRESSION_TEXT_TYPES
    ]
    if trim and words:
        first, last = words[0], words[-1]
        pieces[first] = (pieces[first][0], pieces[first][1].lstrip())
        pieces[last] = (pieces[last][0], pieces[last][1].rstrip())
    return pieces


def _leading_punctuation(text: str) -> str:
    """The punctuation at the very start of ``text``."""
    end = 0
    while end < len(text) and unicodedata.category(text[end]).startswith("P"):
        end += 1
    return text[:end]


def _event_frame(segment: Event | Pause) -> AggregatedTextFrame:
    if isinstance(segment, Pause):
        return AggregatedTextFrame(
            str(segment.duration_ms), TextType.EXPRESSION_PAUSE, raw_text=segment.raw
        )
    return AggregatedTextFrame(segment.name, TextType.EXPRESSION_EVENT, raw_text=segment.raw)
