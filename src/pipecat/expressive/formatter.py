#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reads the expression tags an LLM writes, and formats them for a TTS service.

Part of expressive conversation (alpha). A TTS service with expressive
conversation on is given its text a unit at a time: a sentence, or a token when
it streams tokens. :class:`ExpressionFormatter` handles each unit in three steps:

1. :meth:`~ExpressionFormatter.parse` splits the unit into segments: the words
   to speak, with the delivery they are spoken in, and the sounds and pauses
   between them.
2. :meth:`~ExpressionFormatter.shown_text` gives the words without the tags,
   which clients are shown.
3. :meth:`~ExpressionFormatter.format` builds the request the service is sent,
   in its provider's markup, and says what each stretch of the unit performs,
   for :class:`~pipecat.expressive.spoken.SpokenExpression` to report as the
   words are spoken.

With ElevenLabs' audio tags, one sentence goes::

    "<<cheerful>> It's [[laughs]] ^^back^^ up."       the unit, as the LLM wrote it
    "It's back up."                                   shown to clients
    "[cheerful] It's [laughs] BACK up."               sent to ElevenLabs

The tags are:

- ``<<name ...>>`` sets the delivery the words after it are spoken in, until the
  next delivery tag or the end of the response: ``<<cheerful>>``, or several
  separated by spaces, ``<<cheerful calm>>``. ``<<neutral>>`` returns to the
  voice's baseline. Delivery tags aren't segments of their own; the delivery
  travels with the :class:`Text` it applies to.
- ``<<name: words>>`` is a span: it speaks just those words in its delivery,
  added to the delivery around them, then goes back to it, as in
  ``<<slow: A B 7>>``. ``<<neutral: words>>`` speaks them in the baseline.
- ``^^words^^`` stresses words. It is a span of emphasis, written short, and can
  stand within another span.
- ``[[name]]`` is a :class:`Sound`, a sound at one point (``[[laugh]]``), or a
  :class:`Pause`, ``[[pause N]]`` for N milliseconds.

The brackets decide a tag's kind, except that a name the TTS service declares
keeps its declared kind in the other kind's brackets: ``[[cheerful]]`` still
sets a delivery when ``cheerful`` is a declared delivery. LLMs make that slip,
so it is logged as a warning.

The formatter reads whole units: an aggregator such as
:class:`~pipecat.utils.text.skip_tags_aggregator.SkipTagsAggregator`, given
:data:`EXPRESSION_TAGS`, keeps each tag in one unit as text streams. A tag
left unfinished at the end of a unit is removed, never spoken.

The service's
:class:`~pipecat.adapters.tts.base_expression_adapter.BaseExpressionAdapter`
says what it can perform and formats each piece in its provider's markup.
What the service can't perform is left out of the request, and kept in the
unit's text for the context.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    NEUTRAL_DELIVERY,
    DeliveryScope,
    ExpressionCapabilities,
)
from pipecat.frames.frames import TextType
from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION, StartEndTags

# The delimiters of the tags, for an aggregator such as SkipTagsAggregator to
# keep each tag in one unit.
EXPRESSION_TAGS: list[StartEndTags] = [("<<", ">>"), ("[[", "]]"), ("^^", "^^")]

# The delivery and sound tags in the LLM's text, which word tracking passes over so
# their names aren't matched against spoken words: a span's opening and its
# closing ``>>``, not the words between them, then whole delivery and sound tags.
# Emphasis has no letters in its delimiters, so it needs none.
EXPRESSION_TAG_PATTERN = re.compile(
    r"<<[\w '\-/]+?:[ \t]*(?=[^<>\n]+?>>)|<<[^<>\n]*>>|>>|\[\[[^\[\]\n]*\]\]"
)

# A tag holds words, digits and a little punctuation; anything else is text.
# A span names its delivery before its colon, then the words it delivers.
_TAG = re.compile(
    r"<<([\w '\-/]+?):[ \t]*([^<>\n]+?)>>"
    r"|<<([\w '\-:/]*)>>|\[\[([\w '\-:/]*)\]\]|\^\^([^\n]+?)\^\^"
)
_EMPHASIS = re.compile(r"\^\^([^\n]+?)\^\^")
# What is left of a tag the unit ended in the middle of, and an emphasis that
# never closed.
_UNFINISHED = re.compile(r"(?:<<|\[\[)(?:[\w'\-:/][\w '\-:/]*)?$|<<[\w '\-/]+:[^<>\n]*$")
_UNCLOSED_EMPHASIS = "^^"
_PAUSE = re.compile(r"pause (\d+)")

# Spaces and tabs left side by side where a tag stood between two words.
_SPACES = re.compile(r"[ \t]{2,}")
# Punctuation that closes the words before a tag written ahead of it, as in
# "Hi [[laughs]], there.": what ends a sentence, and what ends a clause.
_CLOSING_PUNCTUATION = re.compile(
    "["
    + re.escape(
        "".join(sorted(SENTENCE_ENDING_PUNCTUATION | {",", ":", ")", "，", "、", "：", "）", "،"}))
    )
    + "]+"
)


@dataclass(frozen=True)
class Text:
    """Words to speak, and the delivery to speak them in.

    ``start`` and ``end`` are offsets into the unit as the LLM wrote it, tags
    included. In ``"<<calm>> Hi."``, the words are
    ``Text(" Hi.", ("calm",), 8, 12)``.

    Parameters:
        text: The words, with the spacing around them.
        delivery: The delivery the words are spoken in, empty for the voice's
            baseline.
        start: Where the words start in the unit.
        end: Where they end.
        span: The delivery a span such as ``<<slow: A B 7>>`` or ``^^B^^`` adds
            for just these words, empty for words outside one.
    """

    text: str
    delivery: tuple[str, ...]
    start: int
    end: int
    span: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sound:
    """A sound made at one point, such as a laugh.

    In ``"Hi [[laughs]]"``, the sound is ``Sound("laughs", 3, 13)``.

    Parameters:
        name: The sound, as written between the brackets.
        start: Where its tag starts in the unit.
        end: Where it ends.
    """

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Pause:
    """A silence at one point, written ``[[pause N]]`` for N milliseconds.

    Parameters:
        duration_ms: How long, in milliseconds.
        start: Where its tag starts in the unit.
        end: Where it ends.
    """

    duration_ms: int
    start: int
    end: int


Segment = Text | Sound | Pause


@dataclass(frozen=True)
class ExpressionSpan:
    """What a stretch of a unit performs: words in their delivery, or a sound.

    :meth:`ExpressionFormatter.format` returns one for each segment the
    service performs, with what it actually performs: a delivery or emphasis the
    service can't perform is left out of ``delivery``, and a sound it can't
    perform has no span. Offsets are into the unit as the LLM wrote it, so
    :class:`~pipecat.expressive.spoken.SpokenExpression` can find which spans
    the text of each word the service reports covers.

    For ``"<<calm>> Hi [[laughs]]"`` on a service that performs both::

        ExpressionSpan(8, 12, delivery=["calm"])
        ExpressionSpan(12, 22, sound=(TextType.EXPRESSION_SOUND, "laughs"))

    Parameters:
        start: Where the stretch starts in the unit.
        end: Where it ends.
        delivery: The delivery its words are spoken in, empty for the voice's
            baseline, or None for a sound.
        sound: The sound or pause, as reported to clients: its text type,
            ``EXPRESSION_SOUND`` or ``EXPRESSION_PAUSE``, and its text, the
            sound's name or the pause's milliseconds. None for words.
    """

    start: int
    end: int
    delivery: list[str] | None = None
    sound: tuple[TextType, str] | None = None


class ExpressionFormatter:
    """Reads a TTS service's expression tags and formats its requests.

    A TTS service has one formatter, which follows a response from unit to
    unit. A delivery set in one sentence carries on into the next, until the next
    delivery tag or :meth:`reset`, which the service calls when the response ends
    or is interrupted. Call :meth:`parse` on each unit in order, then
    :meth:`format` to build its request. What the service can't perform is
    left out of the request, with one warning per tag.

    Example::

        formatter = ExpressionFormatter(lambda: adapter, owner=tts)

        segments = formatter.parse("<<cheerful>> It's [[laughs]] ^^back^^ up.")
        # [Text(" It's ", ("cheerful",), 12, 18),
        #  Sound("laughs", 18, 28),
        #  Text(" ", ("cheerful",), 28, 29),
        #  Text("back", ("cheerful", "emphasis"), 31, 35),
        #  Text(" up.", ("cheerful",), 37, 41)]

        ExpressionFormatter.shown_text(segments, TextType.SENTENCE)
        # "It's back up."

        request, spans = await formatter.format(
            segments, TextType.SENTENCE, transform, trim
        )
        # With ElevenLabs' audio tags: "[cheerful] It's [laughs] BACK up."

        formatter.parse("Bye.")
        # [Text("Bye.", ("cheerful",), 0, 4)]: still cheerful
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
        # The delivery the next words of the response are spoken in.
        self._active_delivery: tuple[str, ...] = ()
        # The delivery of the words last formatted, so markup marks only changes.
        self._formatted_delivery: list[str] | None = None
        self._unperformed: set[str] = set()
        self._delivery: list[str] = []

    @property
    def delivery(self) -> list[str]:
        """The delivery the latest request is spoken in, empty for the voice's baseline.

        It is the delivery its first words are spoken in. A service that
        takes delivery with each request, as a setting rather than as markup in
        the text, sends these.
        """
        return self._delivery

    def capabilities(self) -> ExpressionCapabilities | None:
        """The expression the service can perform now.

        Returns:
            The service's adapter's capabilities, or None when it performs none.
        """
        return self._adapter().capabilities()

    def parse(self, text: str, *, whole: bool = False) -> list[Segment]:
        """Split a unit of text into segments.

        Words take the delivery in effect where they are written, which may have
        been set by an earlier unit of the response::

            formatter.parse("<<calm>> All set.")
            # [Text(" All set.", ("calm",), 8, 17)]
            formatter.parse(" Bye.")
            # [Text(" Bye.", ("calm",), 0, 5)]

        Text that stands on its own, such as a ``TTSSpeakFrame``'s spoken in
        the middle of a response, is parsed with ``whole=True``: it starts in
        the voice's baseline, and the response's delivery carries on after it::

            formatter.parse("<<excited>> One moment.", whole=True)
            # [Text(" One moment.", ("excited",), 11, 23)]
            formatter.parse(" Back.")
            # [Text(" Back.", ("calm",), 0, 6)]

        Args:
            text: The unit, with each of its tags whole.
            whole: Whether the text stands on its own, apart from any
                response.

        Returns:
            Its segments, in order. Spacing alone between two tags is a
            :class:`Text` of its own, so the request keeps it.
        """
        response_delivery = self._active_delivery
        if whole:
            self._active_delivery = ()
        # The names the service declares keep their kind in either brackets.
        capabilities = self.capabilities()
        declared = capabilities or ExpressionCapabilities()
        sounds = {term.name for term in declared.sounds}
        delivery = {NEUTRAL_DELIVERY, *(term.name for term in declared.delivery)}
        segments: list[Segment] = []
        position = 0
        for match in _TAG.finditer(text):
            self._add_text(segments, text, position, match.start())
            span_delivery, span_words, delivery_tag, sound_tag, stressed = match.groups()
            if span_words is not None:
                self._add_span(segments, tuple(span_delivery.split()), span_words, match.start(2))
            elif stressed is not None:
                self._add_span(segments, (EMPHASIS_DELIVERY,), stressed, match.start(5))
            else:
                self._add_tag(segments, delivery_tag, sound_tag, match, sounds, delivery)
            position = match.end()
        self._add_text(segments, text, position, len(text))
        if whole:
            self._active_delivery = response_delivery
        return segments

    def restate_delivery(self):
        """Have the next request state its delivery from its start.

        Markup marks only where the delivery changes, so a request continuing
        the one before it doesn't repeat them. Text spoken in an audio context
        of its own, such as a ``TTSSpeakFrame``'s, follows no earlier request
        there, nor does the response's next request follow it, so the
        service calls this before and after it.
        """
        self._formatted_delivery = None

    def reset(self):
        """Return to the baseline, as at the end of a response or an interruption."""
        self._active_delivery = ()
        self._formatted_delivery = None

    @staticmethod
    def shown_text(segments: list[Segment], text_type: TextType | str) -> str:
        """The words of a unit, without its tags, spaced as written.

        This is the unit's text as clients are shown it::

            segments = formatter.parse("<<cheerful>> It's [[laughs]] ^^back^^ up.")
            ExpressionFormatter.shown_text(segments, TextType.SENTENCE)
            # "It's back up."

        A tag is removed as a word, with the space before it when punctuation
        follows it: ``"Hi [[laughs]], there."`` is shown as ``"Hi, there."``,
        and so is ``"Hi <<calm>>, there."``.

        Args:
            segments: The unit's segments.
            text_type: The unit's text type. A sentence's words are trimmed at
                both ends; a streamed token keeps its spacing.

        Returns:
            The words, with one space where a tag stood between two.
        """
        words = ""
        tag_since_words = False
        end = None
        for segment in segments:
            if not isinstance(segment, Text):
                tag_since_words = bool(words)
                continue
            # A gap between two runs of words is where a tag stood.
            if end is not None and segment.start > end:
                tag_since_words = bool(words)
            # A tag is removed as a word, with the space before it when
            # punctuation follows it.
            if tag_since_words and _CLOSING_PUNCTUATION.match(segment.text):
                words = words.rstrip()
            words += segment.text
            tag_since_words = False
            end = segment.end
        words = _SPACES.sub(" ", words)
        return words if text_type == TextType.TOKEN else words.strip()

    async def format(
        self,
        segments: list[Segment],
        text_type: TextType | str,
        transform: Callable[[str, TextType | str], Awaitable[str | None]],
        trim: Callable[[str], str | None],
    ) -> tuple[str, list[ExpressionSpan]]:
        """Build the request for a unit.

        Each run of words goes through ``transform`` first, so text
        transforms such as markdown stripping never see markup, and is then
        formatted by the adapter in the delivery the service can perform. Sounds
        and pauses are formatted at their places, except that punctuation
        written after one stays with the words before it, where the service
        reports it: ``"Hi [[laughs]], there."`` asks for ``"Hi, [laughs]
        there."``. A sentence's request is trimmed at both ends; a streamed
        token keeps its spacing.

        Cartesia has no emphasis, and changes emotion only between sentences::

            segments = formatter.parse("<<happy>> It's [[laughter]] ^^back^^ up.")
            request, spans = await formatter.format(
                segments, TextType.SENTENCE, transform, trim
            )
            # request: '<emotion value="happy"/> It's [laughter] back up.'
            # spans:   [ExpressionSpan(9, 15, delivery=["happy"]),
            #           ExpressionSpan(15, 27, sound=(EXPRESSION_SOUND, "laughter")),
            #           ExpressionSpan(27, 28, delivery=["happy"]),
            #           ExpressionSpan(30, 34, delivery=["happy"]),  # no emphasis
            #           ExpressionSpan(36, 40, delivery=["happy"])]

        Args:
            segments: The unit's segments.
            text_type: The unit's text type.
            transform: Rewrites a run of words for the service, or returns
                None when nothing is left of it.
            trim: Trims the unit's first words as the service trims a unit's
                text, or returns None when nothing is left of them.

        Returns:
            The text to send to the service, empty when nothing is left to
            say, and a span for each stretch of the unit the service performs.
        """
        capabilities = self.capabilities()
        if text_type != TextType.TOKEN:
            # A sentence's request states its delivery from its start, since not
            # every service keeps delivery from one request to the next.
            self._formatted_delivery = None
        parts: list[str] = []
        spans: list[ExpressionSpan] = []
        first_delivery: list[str] | None = None
        trimmed = False
        # Where the request's last words are, whether a tag was written after
        # them, and where the last run of words ends in the unit.
        words_at: int | None = None
        tag_since_words = False
        text_end: int | None = None
        # The delivery of the request's last words.
        last_delivery: list[str] | None = None
        sentence_delivery = None
        if capabilities and capabilities.delivery_scope == DeliveryScope.SENTENCE:
            sentence_delivery = self._sentence_delivery(segments, capabilities)
        for segment in segments:
            if isinstance(segment, Text):
                delivery = self._performed_delivery(list(segment.delivery), capabilities)
                if sentence_delivery is not None:
                    delivery = sentence_delivery
                if first_delivery is None and segment.text.strip():
                    first_delivery = delivery
                spans.append(ExpressionSpan(segment.start, segment.end, delivery=delivery))
                # A gap between two runs of words is where a tag stood.
                if text_end is not None and segment.start > text_end:
                    tag_since_words = True
                text_end = segment.end
                if not segment.text.strip():
                    parts.append(segment.text)
                    continue
                words_text: str | None = segment.text
                closing = _CLOSING_PUNCTUATION.match(segment.text)
                if closing and words_at is not None and tag_since_words:
                    parts[words_at] = parts[words_at].rstrip() + closing.group()
                    words_text = segment.text[closing.end() :] or None
                tag_since_words = False
                if not trimmed and words_text is not None:
                    # The unit's first words are trimmed as its text would be, and
                    # a sentence's even after a sound or pause written before them.
                    first = words_text if text_type == TextType.TOKEN else words_text.lstrip()
                    words_text = trim(first)
                    trimmed = True
                words = await transform(words_text, text_type) if words_text is not None else None
                if words is not None:
                    words_at = len(parts)
                    parts.append(self._format(words, delivery))
                    last_delivery = delivery
            else:
                tag_since_words = True
                if _performs(segment, capabilities):
                    sound = _sound_text(segment, capabilities)
                    spans.append(ExpressionSpan(segment.start, segment.end, sound=sound))
                    parts.append(self._format_sound(segment, capabilities))
                else:
                    self._warn_unperformed(_tag(segment))
        if first_delivery is not None:
            self._delivery = first_delivery
        if last_delivery is not None and text_type != TextType.TOKEN:
            parts.append(self._adapter().format_end(last_delivery))
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
            segments.append(Text(words, self._active_delivery, start, end))

    def _sentence_delivery(
        self, segments: list[Segment], capabilities: ExpressionCapabilities
    ) -> list[str] | None:
        """The one delivery a request takes on a service that changes delivery between requests.

        It is the delivery its words start in, and any span's, which such a
        service can perform only over the whole request: ``"Your code is
        <<slow: A B 7>>."`` is spoken slowly. Emphasis is the exception, as
        stress over a whole sentence stresses nothing. Clients are told the
        delivery that was requested.
        """
        words = [s for s in segments if isinstance(s, Text) and s.text.strip()]
        if not words:
            return None
        delivery = list(words[0].delivery)
        for segment in words:
            spread = (t for t in segment.span if t != EMPHASIS_DELIVERY and t not in delivery)
            delivery += spread
        return self._performed_delivery(delivery, capabilities)

    def _add_span(
        self,
        segments: list[Segment],
        span: tuple[str, ...],
        text: str,
        start: int,
        around: tuple[str, ...] | None = None,
    ):
        """Add a span's words, in the delivery around it and its own, or in the baseline for neutral.

        ``^^B^^`` is a span of emphasis, written short, and can stand within
        another span: ``<<slow: A ^^B^^ 7>>`` stresses B on top of slow.
        """
        around = self._active_delivery if around is None else around
        if span == (NEUTRAL_DELIVERY,):
            delivery, span = (), ()
        else:
            delivery = (*around, *(t for t in span if t not in around))
        position = 0
        for match in _EMPHASIS.finditer(text):
            if match.start() > position:
                words = text[position : match.start()]
                segments.append(
                    Text(words, delivery, start + position, start + match.start(), span)
                )
            stressed = (*span, EMPHASIS_DELIVERY)
            self._add_span(segments, stressed, match.group(1), start + match.start(1), delivery)
            position = match.end()
        if position < len(text):
            segments.append(
                Text(text[position:], delivery, start + position, start + len(text), span)
            )

    def _add_tag(
        self,
        segments: list[Segment],
        delivery_tag: str | None,
        sound_tag: str | None,
        match: re.Match[str],
        sounds: set[str],
        delivery: set[str],
    ):
        content = (delivery_tag if delivery_tag is not None else sound_tag or "").strip()
        if delivery_tag is None:
            pause = _PAUSE.fullmatch(content)
            if pause:
                segments.append(Pause(int(pause.group(1)), match.start(), match.end()))
                return
            if content in delivery and content not in sounds:
                logger.warning(
                    f"Reading a delivery written in sound brackets as a delivery: {match[0]!r}"
                )
                self._set_delivery(content)
                return
            segments.append(Sound(content, match.start(), match.end()))
            return
        if content in sounds and content not in delivery:
            logger.warning(f"Reading a sound written in delivery brackets as a sound: {match[0]!r}")
            segments.append(Sound(content, match.start(), match.end()))
            return
        self._set_delivery(content)

    def _set_delivery(self, content: str):
        delivery = tuple(content.split())
        self._active_delivery = () if delivery == (NEUTRAL_DELIVERY,) else delivery

    def _format(self, text: str, delivery: list[str]) -> str:
        """Format words to send to the TTS in their delivery."""
        # Leading spacing stays outside the markup, between the words.
        words = text.lstrip()
        formatted = self._adapter().format_text(words, delivery, self._formatted_delivery)
        self._formatted_delivery = delivery
        return text[: len(text) - len(words)] + formatted

    def _performed_delivery(
        self, delivery: list[str], capabilities: ExpressionCapabilities | None
    ) -> list[str]:
        """The delivery the service can perform, warning once about each it can't."""
        performed = []
        for name in delivery:
            if name == EMPHASIS_DELIVERY:
                supported = bool(capabilities and capabilities.emphasis)
            else:
                supported = bool(
                    capabilities
                    and (
                        capabilities.free_text_delivery
                        or any(term.name == name for term in capabilities.delivery)
                    )
                )
            if supported:
                performed.append(name)
            else:
                self._warn_unperformed(
                    "^^emphasis^^" if name == EMPHASIS_DELIVERY else f"<<{name}>>"
                )
        return performed

    def _format_sound(
        self, segment: Sound | Pause, capabilities: ExpressionCapabilities | None
    ) -> str:
        """Format a sound or pause the service can perform."""
        if isinstance(segment, Pause):
            assert capabilities
            return self._adapter().format_pause(_pause_ms(segment, capabilities))
        return self._adapter().format_sound(segment.name)

    def _warn_unperformed(self, tag: str):
        if tag not in self._unperformed:
            self._unperformed.add(tag)
            logger.warning(f"{self._owner} can't perform {tag}, leaving it out of its speech")


def _performs(segment: Sound | Pause, capabilities: ExpressionCapabilities | None) -> bool:
    """Whether the service can perform a sound or pause."""
    if not capabilities:
        return False
    if isinstance(segment, Pause):
        return capabilities.pause_range_ms is not None
    return capabilities.free_text_sounds or any(
        term.name == segment.name for term in capabilities.sounds
    )


def _pause_ms(pause: Pause, capabilities: ExpressionCapabilities) -> int:
    """A pause's length, within the service's range."""
    assert capabilities.pause_range_ms
    shortest, longest = capabilities.pause_range_ms
    return min(max(pause.duration_ms, shortest), longest)


def _sound_text(
    segment: Sound | Pause, capabilities: ExpressionCapabilities | None
) -> tuple[TextType, str]:
    """A sound or pause as reported to clients, with its text type."""
    if isinstance(segment, Pause):
        assert capabilities
        return TextType.EXPRESSION_PAUSE, str(_pause_ms(segment, capabilities))
    return TextType.EXPRESSION_SOUND, segment.name


def _tag(segment: Sound | Pause) -> str:
    """A sound or pause written as a tag."""
    if isinstance(segment, Pause):
        return f"[[pause {segment.duration_ms}]]"
    return f"[[{segment.name}]]"
