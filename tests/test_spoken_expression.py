#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for telling the expression of the words a TTS service reports speaking."""

from pipecat.expressive.formatter import ExpressionSpan
from pipecat.expressive.spoken import SpokenExpression
from pipecat.frames.frames import TextType

LAUGH = (TextType.EXPRESSION_EVENT, "laugh")

TEXT = "<<calm>> Hi. [[laugh]] Bye."
SPANS = [
    ExpressionSpan(8, 13, modes=["calm"]),
    ExpressionSpan(13, 22, event=LAUGH),
    ExpressionSpan(22, 27, modes=["calm", "emphasis"]),
]


def spoken_expression() -> SpokenExpression:
    spoken = SpokenExpression()
    spoken.add(TEXT, SPANS)
    return spoken


def test_a_word_is_given_its_modes_and_the_events_before_it():
    spoken = spoken_expression()

    assert spoken.spoken("Hi.") == (["calm"], [])
    assert spoken.spoken("[[laugh]] Bye.") == (["calm", "emphasis"], [LAUGH])
    assert spoken.modes == ["calm", "emphasis"]


def test_units_added_follow_one_another():
    spoken = spoken_expression()
    spoken.add(" Again.", [ExpressionSpan(0, 7, modes=[])])

    spoken.spoken("Bye.")

    assert spoken.spoken("Again.") == ([], [])


def test_spoken_markup_stands_for_the_next_event():
    spoken = spoken_expression()
    spoken.spoken("Hi.")

    assert spoken.spoken_markup() == LAUGH
    # Reported once, by its markup.
    assert spoken.spoken("[[laugh]] Bye.") == (["calm", "emphasis"], [])


def test_spoken_markup_with_no_event_before_the_next_words_isnt_spoken():
    assert spoken_expression().spoken_markup() is None


def test_text_spoken_before_any_was_added_has_no_modes():
    spoken = SpokenExpression()

    assert spoken.spoken("Hi.") == (None, [])
    assert spoken.modes is None
