"""Sentencex adapter offset and incremental aggregation contracts."""

import pytest

from pipecat.utils.string import match_endofsentence


@pytest.mark.parametrize(
    ("text", "prefix"),
    [
        ("Hello.   Next", "Hello."),
        ("Hello.\n\nNext", "Hello."),
        ("\n\nHello. Next", "\n\nHello."),
        ("👋 Hello. 😀 Next", "👋 Hello."),
        ("Café is open. Next", "Café is open."),
    ],
)
def test_source_character_offsets(text, prefix):
    boundary = match_endofsentence(text)
    assert boundary == len(prefix)
    assert text[:boundary] == prefix
