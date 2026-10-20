#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the expression tag parser."""

import pytest

from pipecat.expressive.parser import (
    Emphasis,
    Event,
    ExpressionParser,
    Mode,
    Pause,
    Segment,
    Text,
)


def merged(segments: list[Segment]) -> list[Segment]:
    """Join neighbouring Text segments, which streaming splits wherever a chunk ends."""
    result: list[Segment] = []
    for segment in segments:
        if isinstance(segment, Text) and result and isinstance(result[-1], Text):
            result[-1] = Text(result[-1].text + segment.text)
        else:
            result.append(segment)
    return result


def parse(text: str, **kwargs) -> list[Segment]:
    parser = ExpressionParser(**kwargs)
    return merged(parser.feed(text) + parser.flush())


def parse_in_pieces(pieces: list[str], **kwargs) -> list[Segment]:
    parser = ExpressionParser(**kwargs)
    segments: list[Segment] = []
    for piece in pieces:
        segments += parser.feed(piece)
    return merged(segments + parser.flush())


REPLY = (
    "<<apologetic>> I'm so sorry about the outage. <<enunciate>> Your ticket is 2, 3, 8, 7. "
    "[[pause 600]] <<cheerful>> The good news is it's *back up* now. [[laugh]] Finally!"
)


def test_a_reply_splits_into_words_and_tags_in_order():
    assert parse(REPLY) == [
        Mode("apologetic", "<<apologetic>>"),
        Text(" I'm so sorry about the outage. "),
        Mode("enunciate", "<<enunciate>>"),
        Text(" Your ticket is 2, 3, 8, 7. "),
        Pause(600, "[[pause 600]]"),
        Text(" "),
        Mode("cheerful", "<<cheerful>>"),
        Text(" The good news is it's "),
        Emphasis("back up", "*back up*"),
        Text(" now. "),
        Event("laugh", "[[laugh]]"),
        Text(" Finally!"),
    ]


STREAMED = [
    REPLY,
    "I'm sorry [about] that. <cheerful> Bye [[laugh]] a < b.",
    "It's **fixed**, *really*. * item 3 * 4 a*b*c <</warm>> [[ Pause  300 ]]",
]


@pytest.mark.parametrize("text", STREAMED)
def test_streaming_gives_the_same_result_wherever_the_text_is_cut(text):
    whole = parse(text, modes=["cheerful"])
    for cut in range(1, len(text)):
        assert parse_in_pieces([text[:cut], text[cut:]], modes=["cheerful"]) == whole, cut


@pytest.mark.parametrize("text", STREAMED)
def test_streaming_one_character_at_a_time(text):
    assert parse_in_pieces(list(text), modes=["cheerful"]) == parse(text, modes=["cheerful"])


def test_text_without_tags_passes_through():
    assert parse("Hello there. How are you?") == [Text("Hello there. How are you?")]


def test_a_pause_is_a_duration_in_milliseconds():
    assert parse("[[pause 1500]]") == [Pause(1500, "[[pause 1500]]")]


@pytest.mark.parametrize(
    "tag,segment",
    [
        ("[[pause]]", Event("pause", "[[pause]]")),
        ("[[pause 800ms]]", Event("pause 800ms", "[[pause 800ms]]")),
        ("<<pause 500>>", Mode("pause 500", "<<pause 500>>")),
        ("<< Cheerful >>", Mode(" Cheerful ", "<< Cheerful >>")),
        ("<</cheerful>>", Mode("/cheerful", "<</cheerful>>")),
        ("[[]]", Event("", "[[]]")),
    ],
)
def test_anything_else_between_the_brackets_is_passed_on_as_written(tag, segment):
    assert parse(tag) == [segment]


def test_unknown_names_follow_their_brackets():
    assert parse("[[snort]] <<smug>>") == [
        Event("snort", "[[snort]]"),
        Text(" "),
        Mode("smug", "<<smug>>"),
    ]


def test_a_known_mode_in_event_brackets_stays_a_mode():
    assert parse("[[cheerful]] Hi", modes=["cheerful"]) == [
        Mode("cheerful", "[[cheerful]]"),
        Text(" Hi"),
    ]


def test_a_known_event_in_mode_brackets_stays_an_event():
    assert parse("<<laugh>> Hi", events=["laugh"]) == [Event("laugh", "<<laugh>>"), Text(" Hi")]


def test_neutral_is_always_a_mode():
    assert parse("[[neutral]] Hi") == [Mode("neutral", "[[neutral]]"), Text(" Hi")]


@pytest.mark.parametrize("text", ["He said [sic].", "Hi [cheerful] there.", "a < b", "a <b> c"])
def test_single_brackets_are_text(text):
    assert parse(text, modes=["cheerful"]) == [Text(text)]


def test_a_tag_left_unfinished_is_removed():
    assert parse("Thanks for calling. <<chee") == [Text("Thanks for calling. ")]
    assert parse("Thanks for calling. [[") == [Text("Thanks for calling. ")]


def test_reset_drops_a_tag_cut_off_by_an_interruption():
    parser = ExpressionParser()
    assert parser.feed("Okay. [[la") == [Text("Okay. ")]
    parser.reset()
    assert parser.feed("Hello") == [Text("Hello")]


@pytest.mark.parametrize(
    "text",
    [
        "Shift it left: a << b.",
        "Brackets [[like this.]] stay text.",
        "Ends with a shift <<",
    ],
)
def test_text_that_only_looks_like_a_tag_is_spoken(text):
    expected = [Text(text)] if not text.endswith("<<") else [Text(text[:-2])]
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text,emphasis",
    [
        ("It's *fixed* now.", Emphasis("fixed", "*fixed*")),
        ("It's **fixed** now.", Emphasis("fixed", "**fixed**")),
        ("It's *back up*.", Emphasis("back up", "*back up*")),
    ],
)
def test_emphasis(text, emphasis):
    assert emphasis in parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "* first item",
        "Three * four is twelve.",
        "2*3*4 is twenty-four.",
        "An *unclosed span",
        "A * spaced * span",
        "Snake_case and *.txt files",
    ],
)
def test_asterisks_that_are_not_emphasis_are_text(text):
    assert parse(text) == [Text(text)]


def test_emphasis_is_held_until_it_closes():
    parser = ExpressionParser()
    assert parser.feed("It's *back") == [Text("It's ")]
    assert parser.feed(" up* now") == [Emphasis("back up", "*back up*"), Text(" now")]
