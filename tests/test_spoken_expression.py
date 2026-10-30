#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for telling the expression of the words a TTS service reports speaking."""

from pipecat.expressive.formatter import ExpressionSpan
from pipecat.expressive.spoken import SpokenExpression
from pipecat.frames.frames import TextType

LAUGH = (TextType.EXPRESSION_SOUND, "laugh")

TEXT = "<<calm>> Hi. [[laugh]] Bye."
SPANS = [
    ExpressionSpan(8, 13, delivery=["calm"]),
    ExpressionSpan(13, 22, sound=LAUGH),
    ExpressionSpan(22, 27, delivery=["calm", "emphasis"]),
]


def spoken_expression() -> SpokenExpression:
    spoken = SpokenExpression()
    spoken.add(TEXT, SPANS)
    return spoken


def test_a_word_is_given_its_delivery_and_the_sounds_before_it():
    spoken = spoken_expression()

    assert spoken.spoken("Hi.") == (["calm"], [])
    assert spoken.spoken("[[laugh]] Bye.") == (["calm", "emphasis"], [LAUGH])
    assert spoken.delivery == ["calm", "emphasis"]


def test_units_added_follow_one_another():
    spoken = spoken_expression()
    spoken.add(" Again.", [ExpressionSpan(0, 7, delivery=[])])

    spoken.spoken("Bye.")

    assert spoken.spoken("Again.") == ([], [])


def test_spoken_markup_stands_for_the_next_sound():
    spoken = spoken_expression()
    spoken.spoken("Hi.")

    assert spoken.spoken_markup() == LAUGH
    # Reported once, by its markup.
    assert spoken.spoken("[[laugh]] Bye.") == (["calm", "emphasis"], [])


def test_spoken_markup_with_no_sound_before_the_next_words_isnt_spoken():
    assert spoken_expression().spoken_markup() is None


def test_a_sound_after_the_last_word_is_left_for_what_follows():
    spoken = SpokenExpression()
    # "Bye [[laugh]]!", spoken as one word that covers the tag.
    spoken.add(
        "Bye [[laugh]]!", [ExpressionSpan(0, 4, delivery=[]), ExpressionSpan(4, 13, sound=LAUGH)]
    )

    assert spoken.spoken("Bye [[laugh]]!") == ([], [])
    assert spoken.spoken_markup() == LAUGH


def test_sounds_no_text_reported_are_unreported_at_the_end():
    spoken = SpokenExpression()
    spoken.add(
        "Bye [[laugh]]", [ExpressionSpan(0, 4, delivery=[]), ExpressionSpan(4, 13, sound=LAUGH)]
    )

    spoken.spoken("Bye")

    assert spoken.unreported() == [LAUGH]
    assert spoken.unreported() == []


def test_text_spoken_before_any_was_added_has_no_delivery():
    spoken = SpokenExpression()

    assert spoken.spoken("Hi.") == (None, [])
    assert spoken.delivery is None
