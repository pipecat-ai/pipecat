#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the expression tag parser."""

import pytest

from pipecat.expressive.parser import Event, ExpressionParser, Pause, Segment, Text


def merged(segments: list[Segment]) -> list[Segment]:
    """Join neighbouring Text in the same modes, which streaming splits wherever a chunk ends."""
    result: list[Segment] = []
    for segment in segments:
        last = result[-1] if result else None
        if isinstance(segment, Text) and isinstance(last, Text) and last.modes == segment.modes:
            result[-1] = Text(last.text + segment.text, last.modes, last.raw + segment.raw)
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


def test_a_reply_splits_into_words_with_their_modes_and_events():
    assert parse(REPLY) == [
        Text(
            " I'm so sorry about the outage. ",
            ("apologetic",),
            "<<apologetic>> I'm so sorry about the outage. ",
        ),
        Text(
            " Your ticket is 2, 3, 8, 7. ",
            ("enunciate",),
            "<<enunciate>> Your ticket is 2, 3, 8, 7. ",
        ),
        Pause(600, "[[pause 600]]"),
        Text(" ", ("enunciate",), " "),
        Text(" The good news is it's ", ("cheerful",), "<<cheerful>> The good news is it's "),
        Text("back up", ("cheerful", "emphasis"), "*back up*"),
        Text(" now. ", ("cheerful",), " now. "),
        Event("laugh", "[[laugh]]"),
        Text(" Finally!", ("cheerful",), " Finally!"),
    ]


def test_the_raw_text_joins_up_to_what_the_llm_wrote():
    assert "".join(s.raw for s in parse(REPLY)) == REPLY


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


def test_text_without_tags_is_in_the_baseline():
    assert parse("Hello there. How are you?") == [
        Text("Hello there. How are you?", (), "Hello there. How are you?")
    ]


def test_several_modes_are_separated_by_spaces():
    assert parse("<<cheerful calm>>Hi") == [Text("Hi", ("cheerful", "calm"), "<<cheerful calm>>Hi")]


def test_a_mode_lasts_until_the_next_mode_tag():
    assert parse("<<calm>>One. <<warm>>Two. Three.") == [
        Text("One. ", ("calm",), "<<calm>>One. "),
        Text("Two. Three.", ("warm",), "<<warm>>Two. Three."),
    ]


def test_neutral_returns_to_the_baseline():
    assert parse("<<calm>>One. <<neutral>>Two.") == [
        Text("One. ", ("calm",), "<<calm>>One. "),
        Text("Two.", (), "<<neutral>>Two."),
    ]


def test_emphasis_is_added_to_the_active_modes():
    assert parse("<<calm>>It's *fixed* now.") == [
        Text("It's ", ("calm",), "<<calm>>It's "),
        Text("fixed", ("calm", "emphasis"), "*fixed*"),
        Text(" now.", ("calm",), " now."),
    ]


def test_the_modes_end_with_the_response():
    parser = ExpressionParser()
    parser.feed("<<calm>>One.")
    parser.flush()
    assert parser.feed("Two") + parser.flush() == [Text("Two", (), "Two")]


def test_a_mode_tag_with_nothing_after_it_goes_with_the_next_segment():
    assert parse("<<calm>>[[laugh]] Hi") == [
        Event("laugh", "<<calm>>[[laugh]]"),
        Text(" Hi", ("calm",), " Hi"),
    ]


def test_a_mode_tag_ending_the_response_is_dropped():
    assert parse("Bye! <<warm>>") == [Text("Bye! ", (), "Bye! ")]


def test_a_pause_is_a_duration_in_milliseconds():
    assert parse("[[pause 1500]]") == [Pause(1500, "[[pause 1500]]")]


@pytest.mark.parametrize(
    "text,segment",
    [
        ("[[pause]]x", Event("pause", "[[pause]]")),
        ("[[pause 800ms]]x", Event("pause 800ms", "[[pause 800ms]]")),
        ("[[clear throat]]x", Event("clear throat", "[[clear throat]]")),
        ("<<pause 500>>x", Text("x", ("pause", "500"), "<<pause 500>>x")),
        ("<</cheerful>>x", Text("x", ("/cheerful",), "<</cheerful>>x")),
    ],
)
def test_anything_else_between_the_brackets_is_passed_on_as_written(text, segment):
    assert parse(text)[0] == segment


def test_a_declared_mode_in_event_brackets_sets_the_mode():
    assert parse("[[cheerful]] Hi", modes=["cheerful"]) == [
        Text(" Hi", ("cheerful",), "[[cheerful]] Hi")
    ]


def test_a_declared_event_in_mode_brackets_stays_an_event():
    assert parse("<<laugh>> Hi", events=["laugh"]) == [
        Event("laugh", "<<laugh>>"),
        Text(" Hi", (), " Hi"),
    ]


@pytest.mark.parametrize("text", ["He said [sic].", "Hi [cheerful] there.", "a < b", "a <b> c"])
def test_single_brackets_are_text(text):
    assert parse(text, modes=["cheerful"]) == [Text(text, (), text)]


def test_a_tag_left_unfinished_is_removed():
    assert parse("Thanks for calling. <<chee") == [
        Text("Thanks for calling. ", (), "Thanks for calling. ")
    ]
    assert parse("Thanks for calling. [[") == [
        Text("Thanks for calling. ", (), "Thanks for calling. ")
    ]


def test_reset_drops_a_tag_cut_off_by_an_interruption():
    parser = ExpressionParser()
    assert parser.feed("<<calm>>Okay. [[la") == [Text("Okay. ", ("calm",), "<<calm>>Okay. ")]
    parser.reset()
    assert parser.feed("Hello") == [Text("Hello", (), "Hello")]


@pytest.mark.parametrize(
    "text",
    [
        "Shift it left: a << b.",
        "Brackets [[like this.]] stay text.",
    ],
)
def test_text_that_only_looks_like_a_tag_is_spoken(text):
    assert parse(text) == [Text(text, (), text)]


@pytest.mark.parametrize(
    "text,emphasis",
    [
        ("It's *fixed* now.", Text("fixed", ("emphasis",), "*fixed*")),
        ("It's **fixed** now.", Text("fixed", ("emphasis",), "**fixed**")),
        ("It's *back up*.", Text("back up", ("emphasis",), "*back up*")),
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
    assert parse(text) == [Text(text, (), text)]


def test_emphasis_is_held_until_it_closes():
    parser = ExpressionParser()
    assert parser.feed("It's *back") == [Text("It's ", (), "It's ")]
    assert parser.feed(" up* now") == [
        Text("back up", ("emphasis",), "*back up*"),
        Text(" now", (), " now"),
    ]
