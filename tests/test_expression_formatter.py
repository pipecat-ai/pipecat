#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for reading expression tags and formatting them for a TTS service."""

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.formatter import (
    EXPRESSION_TAGS,
    Event,
    ExpressionFormatter,
    ExpressionSpan,
    Segment,
    Text,
)
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator


class DeclaredAdapter(BaseExpressionAdapter):
    """Declares events and modes, and formats modes as <modes> where they change."""

    def __init__(self, events=(), modes=()):
        self.events = list(events)
        self.modes = list(modes)

    def capabilities(self):
        return ExpressionCapabilities(
            events=[ExpressionTerm(name) for name in self.events],
            modes=[ExpressionTerm(name) for name in self.modes],
            pause_range_ms=(100, 1000),
            emphasis=True,
        )

    def format_text(self, text, modes, previous_modes):
        changed = modes != (previous_modes or [])
        return f"<{','.join(modes)}> {text}" if changed and modes else text

    def format_event(self, name):
        return f" <{name}> "


def formatter(**kwargs) -> ExpressionFormatter:
    adapter = DeclaredAdapter(**kwargs)
    return ExpressionFormatter(lambda: adapter, owner="tts")


def words(text: str, modes: tuple[str, ...]):
    """A Text segment, compared without its position."""
    return ("text", text, modes)


def shape(segment: Segment):
    if isinstance(segment, Text):
        return words(segment.text, segment.modes)
    if isinstance(segment, Event):
        return ("event", segment.name)
    return ("pause", segment.duration_ms)


def parse(text: str, **kwargs) -> list:
    return [shape(s) for s in formatter(**kwargs).parse(text)]


REPLY = (
    "<<apologetic>> I'm so sorry about the outage. <<enunciate>> Your ticket is 2, 3, 8, 7. "
    "[[pause 600]] <<cheerful>> The good news is it's ^^back up^^ now. [[laugh]] Finally!"
)


def test_a_reply_splits_into_words_with_their_modes_and_events():
    assert parse(REPLY) == [
        words(" I'm so sorry about the outage. ", ("apologetic",)),
        words(" Your ticket is 2, 3, 8, 7. ", ("enunciate",)),
        ("pause", 600),
        words(" ", ("enunciate",)),
        words(" The good news is it's ", ("cheerful",)),
        words("back up", ("cheerful", "emphasis")),
        words(" now. ", ("cheerful",)),
        ("event", "laugh"),
        words(" Finally!", ("cheerful",)),
    ]


def test_each_segment_says_where_it_was_written():
    for segment in formatter().parse(REPLY):
        written = REPLY[segment.start : segment.end]
        if isinstance(segment, Text):
            assert written == segment.text
        else:
            assert written.startswith("[[") and written.endswith("]]")


def test_text_without_tags_is_in_the_baseline():
    assert parse("Hello there. How are you?") == [words("Hello there. How are you?", ())]


def test_several_modes_are_separated_by_spaces():
    assert parse("<<cheerful calm>>Hi") == [words("Hi", ("cheerful", "calm"))]


def test_a_mode_lasts_until_the_next_mode_tag_across_units():
    parser = formatter()

    assert [shape(s) for s in parser.parse("<<calm>>One. ")] == [words("One. ", ("calm",))]
    assert [shape(s) for s in parser.parse("Two. <<warm>>Three.")] == [
        words("Two. ", ("calm",)),
        words("Three.", ("warm",)),
    ]


def test_neutral_returns_to_the_baseline():
    assert parse("<<calm>>One. <<neutral>>Two.") == [words("One. ", ("calm",)), words("Two.", ())]


def test_the_modes_end_with_the_response():
    parser = formatter()
    parser.parse("<<calm>>One.")
    parser.reset()

    assert [shape(s) for s in parser.parse("Two")] == [words("Two", ())]


def test_text_that_stands_on_its_own_leaves_the_responses_modes():
    parser = formatter()
    parser.parse("<<calm>>One. ")

    assert [shape(s) for s in parser.parse("Hello.", whole=True)] == [words("Hello.", ())]
    assert [shape(s) for s in parser.parse("Two.")] == [words("Two.", ("calm",))]


def test_a_mode_tag_ending_the_text_says_nothing():
    assert parse("Bye! <<warm>>") == [words("Bye! ", ())]


def test_a_pause_is_a_duration_in_milliseconds():
    assert parse("[[pause 1500]]") == [("pause", 1500)]


@pytest.mark.parametrize(
    "text,segment",
    [
        ("[[pause]]x", ("event", "pause")),
        ("[[pause 800ms]]x", ("event", "pause 800ms")),
        ("[[clear throat]]x", ("event", "clear throat")),
        ("<<pause 500>>x", words("x", ("pause", "500"))),
        ("<</cheerful>>x", words("x", ("/cheerful",))),
    ],
)
def test_anything_else_between_the_brackets_is_passed_on_as_written(text, segment):
    assert parse(text)[0] == segment


def test_a_declared_mode_in_event_brackets_sets_the_mode():
    assert parse("[[cheerful]] Hi", modes=["cheerful"]) == [words(" Hi", ("cheerful",))]


def test_a_declared_event_in_mode_brackets_stays_an_event():
    assert parse("<<laugh>> Hi", events=["laugh"]) == [("event", "laugh"), words(" Hi", ())]


@pytest.mark.parametrize(
    "text",
    [
        "He said [sic].",
        "Hi [cheerful] there.",
        "a < b",
        "a <b> c",
        "Shift it left: a << b.",
        "Brackets [[like this.]] stay text.",
    ],
)
def test_text_that_only_looks_like_a_tag_is_spoken(text):
    assert parse(text, modes=["cheerful"]) == [words(text, ())]


def test_a_tag_left_unfinished_is_removed():
    assert parse("Thanks for calling. <<chee") == [words("Thanks for calling. ", ())]
    assert parse("Thanks for calling. [[") == [words("Thanks for calling. ", ())]


def test_emphasis_is_added_to_the_active_modes():
    assert parse("<<calm>>It's ^^fixed^^ now.") == [
        words("It's ", ("calm",)),
        words("fixed", ("calm", "emphasis")),
        words(" now.", ("calm",)),
    ]


@pytest.mark.parametrize(
    "text,stressed",
    [
        ("It's ^^back up^^.", "back up"),
        ("The code is ^^8492^^.", "8492"),
        ("^^Wow^^, nice.", "Wow"),
    ],
)
def test_emphasis(text, stressed):
    assert words(stressed, ("emphasis",)) in parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "Two to the tenth is 2^10.",
        "a^b^c",
        "Thanks ^_^ bye",
        "It's ^fixed^ and *done*.",
    ],
)
def test_single_carets_and_asterisks_are_text(text):
    assert parse(text) == [words(text, ())]


def test_emphasis_that_never_closes_is_spoken_plainly():
    assert parse("It's ^^back up") == [words("It's back up", ())]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [TextType.TOKEN, TextType.SENTENCE])
async def test_an_aggregator_keeps_each_tag_in_one_unit(mode):
    aggregator = SkipTagsAggregator(EXPRESSION_TAGS, aggregation_type=mode)
    units = []
    for chunk in ["<<chee", "rful>> It's [[la", "ugh]] ^^back", " up^", "^ now."]:
        units += [a.text async for a in aggregator.aggregate(chunk)]
    flushed = await aggregator.flush()
    units += [flushed.text] if flushed else []

    parser = formatter()
    segments = [shape(s) for unit in units for s in parser.parse(unit)]
    assert ("event", "laugh") in segments
    assert words("back up", ("cheerful", "emphasis")) in segments


def test_names_declared_after_the_formatter_is_made_are_known():
    adapter = DeclaredAdapter()
    expression = ExpressionFormatter(lambda: adapter, owner="tts")
    adapter.events = ["laugh"]

    # A declared event keeps its kind in mode brackets.
    segments = expression.parse("<<laugh>> Hi.")

    assert isinstance(segments[0], Event) and segments[0].name == "laugh"
    assert isinstance(segments[1], Text)


def test_a_units_words_are_shown_without_its_tags():
    segments = formatter().parse("<<calm>> It's [[laugh]] fine. [[pause 400]]")

    assert ExpressionFormatter.shown_text(segments, TextType.SENTENCE) == "It's fine."
    assert ExpressionFormatter.shown_text(segments, TextType.TOKEN) == " It's fine. "


async def same(words, text_type):
    return words


@pytest.mark.asyncio
async def test_a_request_formats_what_the_service_performs_and_says_where():
    expression = formatter(events=["laugh"], modes=["calm"])
    text = "<<calm>> Hi. [[laugh]] [[pause 50]]"

    request, spans = await expression.format(
        expression.parse(text), TextType.SENTENCE, same, lambda words: words
    )

    assert request == "<calm> Hi. <laugh>"
    assert spans == [
        ExpressionSpan(8, 13, modes=["calm"]),
        ExpressionSpan(13, 22, event=(TextType.EXPRESSION_EVENT, "laugh")),
        ExpressionSpan(22, 23, modes=["calm"]),
        ExpressionSpan(23, 35, event=(TextType.EXPRESSION_PAUSE, "100")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, shown, sent",
    [
        ("Hi [[laugh]], there.", "Hi, there.", "Hi, <laugh> there."),
        ("It's ^^back^^ [[laugh]]!", "It's back!", "It's <emphasis> back! <laugh>"),
        ("Hi[[laugh]], there.", "Hi, there.", "Hi, <laugh> there."),
        ("Well [[laugh]]... okay.", "Well... okay.", "Well... <laugh> okay."),
        ("Really [[laugh]]？ Yes.", "Really？ Yes.", "Really？ <laugh> Yes."),
        ("مرحبا [[laugh]]؟", "مرحبا؟", "مرحبا؟ <laugh>"),
        # An event the service can't perform is removed the same way.
        ("Hi [[sigh]], there.", "Hi, there.", "Hi, there."),
        # Spacing the LLM wrote before the punctuation is kept.
        ("Hi [[laugh]] , there.", "Hi , there.", "Hi <laugh> , there."),
        ("It's [[laugh]] back.", "It's back.", "It's <laugh> back."),
    ],
)
async def test_punctuation_after_an_event_closes_the_words_before_it(text, shown, sent):
    expression = formatter(events=["laugh"])
    segments = expression.parse(text)

    assert ExpressionFormatter.shown_text(segments, TextType.SENTENCE) == shown
    assert (await expression.format(segments, TextType.SENTENCE, same, lambda w: w))[0] == sent


@pytest.mark.asyncio
async def test_a_request_leaves_out_what_the_service_cant_perform():
    expression = formatter()
    segments = expression.parse("<<calm>> Hi. [[laugh]]", whole=True)

    request, spans = await expression.format(segments, TextType.SENTENCE, same, lambda w: w)

    assert request == "Hi."
    assert spans == [ExpressionSpan(8, 13, modes=[])]
