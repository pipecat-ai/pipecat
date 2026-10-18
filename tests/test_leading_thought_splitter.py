#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LeadingThoughtSplitter."""

from pipecat.utils.text.leading_thought_splitter import (
    LeadingThoughtEvent,
    LeadingThoughtSplitter,
)

START = LeadingThoughtEvent.THOUGHT_START
THOUGHT = LeadingThoughtEvent.THOUGHT_TEXT
END = LeadingThoughtEvent.THOUGHT_END
TEXT = LeadingThoughtEvent.TEXT


def _split(chunks: list[str], flush: bool = True) -> list[tuple[LeadingThoughtEvent, str]]:
    """Feed chunks through a splitter, merging consecutive text of the same kind."""
    splitter = LeadingThoughtSplitter("<think>", "</think>")
    events = [e for chunk in chunks for e in splitter.split(chunk)]
    events += splitter.flush() if flush else splitter.reset()
    merged: list[tuple[LeadingThoughtEvent, str]] = []
    for kind, text in events:
        if merged and kind in (THOUGHT, TEXT) and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


def test_thought_split_across_chunks():
    chunks = ["<th", "ink", ">Let me", " check.</", "think", ">", "Hello", "!"]
    assert _split(chunks) == [
        (START, ""),
        (THOUGHT, "Let me check."),
        (END, ""),
        (TEXT, "Hello!"),
    ]


def test_every_split_point_of_the_tags():
    text = "<think>Hmm.</think>Hi"
    for i in range(1, len(text)):
        assert _split([text[:i], text[i:]]) == [
            (START, ""),
            (THOUGHT, "Hmm."),
            (END, ""),
            (TEXT, "Hi"),
        ], f"split at {i}"


def test_whitespace_before_the_thought_is_allowed():
    assert _split(["\n", "<think>Hmm.</think>Hi"]) == [
        (START, ""),
        (THOUGHT, "Hmm."),
        (END, ""),
        (TEXT, "Hi"),
    ]


def test_whitespace_after_the_thought_is_dropped_across_chunks():
    assert _split(["<think>Hmm.</think>", "\n", "\n ", "Hi there", "\n"]) == [
        (START, ""),
        (THOUGHT, "Hmm."),
        (END, ""),
        (TEXT, "Hi there\n"),
    ]


def test_text_without_a_thought_passes_through_unchanged():
    assert _split([" Hel", "lo <think>", "</think>"]) == [(TEXT, " Hello <think></think>")]


def test_text_that_rules_out_the_tag_is_not_held_back():
    splitter = LeadingThoughtSplitter("<think>", "</think>")
    assert splitter.split("<th") == []
    assert splitter.split("e") == [(TEXT, "<the")]
    assert splitter.split("re") == [(TEXT, "re")]


def test_partial_tag_is_flushed_at_the_end():
    assert _split(["<"]) == [(TEXT, "<")]


def test_thought_left_open_is_closed_at_the_end():
    assert _split(["<think>Hmm", "</thi"]) == [(START, ""), (THOUGHT, "Hmm</thi"), (END, "")]


def test_text_cut_short_is_not_flushed():
    # Inside a thought, the last len("</think>") - 1 characters are held back.
    assert _split(["<think>Let me check.", "</thi"], flush=False) == [
        (START, ""),
        (THOUGHT, "Let me chec"),
        (END, ""),
    ]
    assert _split(["<thi"], flush=False) == []


def test_flush_readies_the_splitter_for_the_next_text():
    splitter = LeadingThoughtSplitter("<thinking>", "</thinking>")
    for text in ["<thinking>One.</thinking>", "<thinking>Two.</thinking>"]:
        events = splitter.split(text) + splitter.flush()
        assert [kind for kind, _ in events] == [START, THOUGHT, END]
    assert splitter.flush() == []
