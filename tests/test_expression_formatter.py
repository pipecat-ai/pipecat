#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for reading expression tags and formatting them for a TTS service."""

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import DeliveryScope, ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.formatter import (
    EXPRESSION_TAG_PATTERN,
    EXPRESSION_TAGS,
    ExpressionFormatter,
    ExpressionSpan,
    Segment,
    Sound,
    Text,
    strip_expression_tags,
)
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator


class DeclaredAdapter(BaseExpressionAdapter):
    """Declares sounds and delivery, and formats delivery as <delivery> where it changes."""

    def __init__(self, sounds=(), delivery=(), delivery_scope=DeliveryScope.ANYWHERE):
        self.sounds = list(sounds)
        self.delivery = list(delivery)
        self.delivery_scope = delivery_scope

    def capabilities(self):
        return ExpressionCapabilities(
            sounds=[ExpressionTerm(name) for name in self.sounds],
            delivery=[ExpressionTerm(name) for name in self.delivery],
            pause_range_ms=(100, 1000),
            emphasis=True,
            delivery_scope=self.delivery_scope,
        )

    def format_text(self, text, delivery, previous_delivery):
        changed = delivery != (previous_delivery or [])
        return f"<{','.join(delivery)}> {text}" if changed and delivery else text

    def format_sound(self, name):
        return f" <{name}> "


def formatter(**kwargs) -> ExpressionFormatter:
    adapter = DeclaredAdapter(**kwargs)
    return ExpressionFormatter(lambda: adapter, owner="tts")


def words(text: str, delivery: tuple[str, ...]):
    """A Text segment, compared without its position."""
    return ("text", text, delivery)


def shape(segment: Segment):
    if isinstance(segment, Text):
        return words(segment.text, segment.delivery)
    if isinstance(segment, Sound):
        return ("sound", segment.name)
    return ("pause", segment.duration_ms)


def parse(text: str, **kwargs) -> list:
    return [shape(s) for s in formatter(**kwargs).parse(text)]


REPLY = (
    "<<apologetic>> I'm so sorry about the outage. <<enunciate>> Your ticket is 2, 3, 8, 7. "
    "[[pause 600]] <<cheerful>> The good news is it's ^^back up^^ now. [[laugh]] Finally!"
)


def test_a_reply_splits_into_words_with_their_delivery_and_sounds():
    assert parse(REPLY) == [
        words(" I'm so sorry about the outage. ", ("apologetic",)),
        words(" Your ticket is 2, 3, 8, 7. ", ("enunciate",)),
        ("pause", 600),
        words(" ", ("enunciate",)),
        words(" The good news is it's ", ("cheerful",)),
        words("back up", ("cheerful", "emphasis")),
        words(" now. ", ("cheerful",)),
        ("sound", "laugh"),
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


def test_a_delivery_name_may_be_several_words():
    assert parse("<<soft  tone>>Hi") == [words("Hi", ("soft tone",))]


def test_a_delivery_lasts_until_the_next_delivery_tag_across_units():
    parser = formatter()

    assert [shape(s) for s in parser.parse("<<calm>>One. ")] == [words("One. ", ("calm",))]
    assert [shape(s) for s in parser.parse("Two. <<warm>>Three.")] == [
        words("Two. ", ("calm",)),
        words("Three.", ("warm",)),
    ]


def test_neutral_returns_to_the_baseline():
    assert parse("<<calm>>One. <<neutral>>Two.") == [words("One. ", ("calm",)), words("Two.", ())]


def test_the_delivery_ends_with_the_response():
    parser = formatter()
    parser.parse("<<calm>>One.")
    parser.reset()

    assert [shape(s) for s in parser.parse("Two")] == [words("Two", ())]


def test_text_that_stands_on_its_own_leaves_the_responses_delivery():
    parser = formatter()
    parser.parse("<<calm>>One. ")

    assert [shape(s) for s in parser.parse("Hello.", whole=True)] == [words("Hello.", ())]
    assert [shape(s) for s in parser.parse("Two.")] == [words("Two.", ("calm",))]


def test_a_delivery_tag_ending_the_text_says_nothing():
    assert parse("Bye! <<warm>>") == [words("Bye! ", ())]


def test_a_pause_is_a_duration_in_milliseconds():
    assert parse("[[pause 1500]]") == [("pause", 1500)]


@pytest.mark.parametrize(
    "text,segment",
    [
        ("[[pause]]x", ("sound", "pause")),
        ("[[pause 800ms]]x", ("sound", "pause 800ms")),
        ("[[clear throat]]x", ("sound", "clear throat")),
        ("<<pause 500>>x", words("x", ("pause 500",))),
        ("<</cheerful>>x", words("x", ("/cheerful",))),
    ],
)
def test_anything_else_between_the_brackets_is_passed_on_as_written(text, segment):
    assert parse(text)[0] == segment


def test_a_declared_delivery_in_sound_brackets_sets_the_delivery():
    assert parse("[[cheerful]] Hi", delivery=["cheerful"]) == [words(" Hi", ("cheerful",))]


def test_a_declared_sound_in_delivery_brackets_stays_a_sound():
    assert parse("<<laugh>> Hi", sounds=["laugh"]) == [("sound", "laugh"), words(" Hi", ())]


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
    assert parse(text, delivery=["cheerful"]) == [words(text, ())]


def test_a_span_adds_its_delivery_to_just_its_words():
    assert parse("<<calm>> Your code is <<slow: A B 7>>. Bye.") == [
        words(" Your code is ", ("calm",)),
        words("A B 7", ("calm", "slow")),
        words(". Bye.", ("calm",)),
    ]


def test_a_neutral_span_speaks_its_words_in_the_baseline():
    assert parse("<<calm>> Say <<neutral: hi there>>, okay.") == [
        words(" Say ", ("calm",)),
        words("hi there", ()),
        words(", okay.", ("calm",)),
    ]


def test_emphasis_in_a_span_is_added_to_its_delivery():
    assert parse("It's <<slow: A ^^B^^ 7>>.") == [
        words("It's ", ()),
        words("A ", ("slow",)),
        words("B", ("slow", "emphasis")),
        words(" 7", ("slow",)),
        words(".", ()),
    ]


def test_emphasis_is_a_span_written_short():
    assert parse("Now ^^that's^^ it.") == parse("Now <<emphasis: that's>> it.")


def test_each_span_segment_says_where_it_was_written():
    text = "<<calm>> Your code is <<slow: A ^^B^^ 7>>, okay."
    for segment in formatter().parse(text):
        assert text[segment.start : segment.end] == segment.text


def test_a_span_left_unfinished_is_removed():
    assert parse("Your code is <<slow: A B") == [words("Your code is ", ())]


def test_word_tracking_passes_over_a_spans_markup_but_not_its_words():
    masked = EXPRESSION_TAG_PATTERN.sub("#", "Code <<slow: A B>>, ok. <<calm>> [[laugh]] Bye.")

    assert masked == "Code #A B#, ok. # # Bye."


def test_a_tag_left_unfinished_is_removed():
    assert parse("Thanks for calling. <<chee") == [words("Thanks for calling. ", ())]
    assert parse("Thanks for calling. [[") == [words("Thanks for calling. ", ())]


def test_emphasis_is_added_to_the_active_delivery():
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
@pytest.mark.parametrize("text_type", [TextType.TOKEN, TextType.SENTENCE])
async def test_an_aggregator_keeps_each_tag_in_one_unit(text_type):
    aggregator = SkipTagsAggregator(EXPRESSION_TAGS, aggregation_type=text_type)
    units = []
    for chunk in ["<<chee", "rful>> It's [[la", "ugh]] ^^back", " up^", "^ now."]:
        units += [a.text async for a in aggregator.aggregate(chunk)]
    flushed = await aggregator.flush()
    units += [flushed.text] if flushed else []

    parser = formatter()
    segments = [shape(s) for unit in units for s in parser.parse(unit)]
    assert ("sound", "laugh") in segments
    assert words("back up", ("cheerful", "emphasis")) in segments


def test_names_declared_after_the_formatter_is_made_are_known():
    adapter = DeclaredAdapter()
    expression = ExpressionFormatter(lambda: adapter, owner="tts")
    adapter.sounds = ["laugh"]

    # A declared sound keeps its kind in delivery brackets.
    segments = expression.parse("<<laugh>> Hi.")

    assert isinstance(segments[0], Sound) and segments[0].name == "laugh"
    assert isinstance(segments[1], Text)


def test_a_units_words_are_shown_without_its_tags():
    segments = formatter().parse("<<calm>> It's [[laugh]] fine. [[pause 400]]")

    assert ExpressionFormatter.shown_text(segments, TextType.SENTENCE) == "It's fine."
    assert ExpressionFormatter.shown_text(segments, TextType.TOKEN) == " It's fine. "


async def same(words, text_type):
    return words


@pytest.mark.asyncio
async def test_a_request_formats_what_the_service_performs_and_says_where():
    expression = formatter(sounds=["laugh"], delivery=["calm"])
    text = "<<calm>> Hi. [[laugh]] [[pause 50]]"

    formatted = await expression.format(
        expression.parse(text), TextType.SENTENCE, same, lambda words: words
    )
    request, spans = formatted.text, formatted.spans

    assert request == "<calm> Hi. <laugh>"
    assert spans == [
        ExpressionSpan(8, 13, delivery=["calm"]),
        ExpressionSpan(13, 22, sound=(TextType.EXPRESSION_SOUND, "laugh")),
        ExpressionSpan(22, 23, delivery=["calm"]),
        ExpressionSpan(23, 35, sound=(TextType.EXPRESSION_PAUSE, "100")),
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
        # A sound the service can't perform is removed the same way.
        ("Hi [[sigh]], there.", "Hi, there.", "Hi, there."),
        # Spacing the LLM wrote before the punctuation is kept.
        ("Hi [[laugh]] , there.", "Hi , there.", "Hi <laugh> , there."),
        ("It's [[laugh]] back.", "It's back.", "It's <laugh> back."),
        # So does punctuation after a span or a delivery tag.
        ("It's <<slow: A B 7>>, okay.", "It's A B 7, okay.", "It's <slow> A B 7, okay."),
        ("Hi <<calm>>, there.", "Hi, there.", "Hi, <calm> there."),
    ],
)
async def test_punctuation_after_a_tag_closes_the_words_before_it(text, shown, sent):
    expression = formatter(sounds=["laugh"], delivery=["calm", "slow"])
    segments = expression.parse(text)

    assert ExpressionFormatter.shown_text(segments, TextType.SENTENCE) == shown
    assert (await expression.format(segments, TextType.SENTENCE, same, lambda w: w)).text == sent


class ClosingAdapter(DeclaredAdapter):
    """Closes each request with the delivery its last words are spoken in."""

    def format_end(self, delivery):
        return f" </{','.join(delivery)}>"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text_type, ends_with",
    [(TextType.SENTENCE, "</calm>"), (TextType.TOKEN, " up.")],
)
async def test_a_sentence_request_is_closed_by_the_adapter(text_type, ends_with):
    adapter = ClosingAdapter(delivery=["calm"])
    expression = ExpressionFormatter(lambda: adapter, owner="tts")

    request = (
        await expression.format(
            expression.parse("<<calm>> It's back up."), text_type, same, lambda w: w
        )
    ).text

    # A streamed token is followed by more of the same request, so isn't closed.
    assert request.endswith(ends_with)


@pytest.mark.asyncio
async def test_a_request_leaves_out_what_the_service_cant_perform():
    expression = formatter()
    segments = expression.parse("<<calm>> Hi. [[laugh]]", whole=True)

    formatted = await expression.format(segments, TextType.SENTENCE, same, lambda w: w)
    request, spans = formatted.text, formatted.spans

    assert request == "Hi."
    assert spans == [ExpressionSpan(8, 13, delivery=[])]


@pytest.mark.asyncio
async def test_a_span_sets_the_delivery_of_its_whole_request_on_a_sentence_service():
    expression = formatter(delivery=["calm", "slow"], delivery_scope=DeliveryScope.SENTENCE)
    segments = expression.parse("<<calm>> Your code is <<slow: A B 7>>, okay.")

    formatted = await expression.format(segments, TextType.SENTENCE, same, lambda w: w)
    request, spans = formatted.text, formatted.spans

    assert request == "<calm,slow> Your code is A B 7, okay."
    assert {tuple(span.delivery) for span in spans} == {("calm", "slow")}


@pytest.mark.asyncio
async def test_emphasis_stays_on_its_words_on_a_sentence_service():
    expression = formatter(delivery=["calm"], delivery_scope=DeliveryScope.SENTENCE)
    segments = expression.parse("<<calm>> It's ^^back^^ now.")

    formatted = await expression.format(segments, TextType.SENTENCE, same, lambda w: w)
    request, spans = formatted.text, formatted.spans

    assert request == "<calm> It's back now."
    assert {tuple(span.delivery) for span in spans} == {("calm",)}


def test_tags_are_stripped_to_the_words_clients_are_shown():
    assert strip_expression_tags("<<cheerful>> It's [[laughs]] ^^back^^ up.") == "It's back up."
    assert strip_expression_tags("Hi [[laughs]], <<calm: there>>.") == "Hi, there."
