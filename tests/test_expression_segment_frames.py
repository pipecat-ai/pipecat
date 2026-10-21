#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the frames a unit of text becomes once its expression tags are read."""

from pipecat.expressive.parser import Event, Pause, Text
from pipecat.expressive.segment_frames import segment_frames
from pipecat.frames.frames import TextType

SEGMENTS = [
    Text(" It's ", ("calm",), "<<calm>> It's "),
    Event("laugh", "[[laugh]]"),
    Text(" fine. ", ("calm",), " fine. "),
    Pause(400, "[[pause 400]]"),
]


def test_words_carry_their_modes_and_events_are_not_spoken_as_words():
    pieces = segment_frames(SEGMENTS, TextType.SENTENCE, trim=True)

    assert [(f.text_type, f.text, f.modes, bool(f.skip_tts)) for f, _ in pieces] == [
        (TextType.SENTENCE, "It's", ["calm"], False),
        (TextType.EXPRESSION_EVENT, "laugh", None, True),
        (TextType.SENTENCE, "fine.", ["calm"], False),
        (TextType.EXPRESSION_PAUSE, "400", None, True),
    ]
    assert [f.raw_text for f, _ in pieces] == [
        "<<calm>> It's",
        "[[laugh]]",
        "fine.",
        "[[pause 400]]",
    ]


def test_the_text_to_speak_keeps_the_spaces_between_the_words():
    pieces = segment_frames(SEGMENTS, TextType.SENTENCE, trim=True)

    assert [text for _, text in pieces] == ["It's ", "", " fine.", ""]


def test_tokens_are_left_as_they_are():
    pieces = segment_frames(SEGMENTS, TextType.TOKEN, trim=False)

    assert [(f.text, text) for f, text in pieces] == [
        (" It's ", " It's "),
        ("laugh", ""),
        (" fine. ", " fine. "),
        ("400", ""),
    ]
