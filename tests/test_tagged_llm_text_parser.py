#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for TaggedLLMTextParser."""

from pipecat.utils.text.tagged_llm_text_parser import (
    TaggedLLMTextParsedOutputChunk,
    TaggedLLMTextParsedOutputChunkKind,
    TaggedLLMTextParser,
)

START = TaggedLLMTextParsedOutputChunkKind.THOUGHT_START
THOUGHT = TaggedLLMTextParsedOutputChunkKind.THOUGHT_TEXT
END = TaggedLLMTextParsedOutputChunkKind.THOUGHT_END
TEXT = TaggedLLMTextParsedOutputChunkKind.TEXT


def _pairs(
    chunks: list[TaggedLLMTextParsedOutputChunk],
) -> list[tuple[TaggedLLMTextParsedOutputChunkKind, str]]:
    return [(chunk.kind, chunk.text) for chunk in chunks]


def _parse(
    texts: list[str], flush: bool = True
) -> list[tuple[TaggedLLMTextParsedOutputChunkKind, str]]:
    """Feed texts through a parser, merging consecutive text of the same kind."""
    parser = TaggedLLMTextParser("<think>", "</think>")
    chunks = [c for text in texts for c in parser.parse_next_chunk(text)]
    chunks += parser.flush() if flush else parser.reset()
    merged: list[tuple[TaggedLLMTextParsedOutputChunkKind, str]] = []
    for kind, text in _pairs(chunks):
        if merged and kind in (THOUGHT, TEXT) and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


def test_thought_split_across_chunks():
    texts = ["<th", "ink", ">Let me", " check.</", "think", ">", "Hello", "!"]
    assert _parse(texts) == [
        (START, ""),
        (THOUGHT, "Let me check."),
        (END, ""),
        (TEXT, "Hello!"),
    ]


def test_every_split_point_of_the_tags():
    text = "<think>Hmm.</think>Hi"
    for i in range(1, len(text)):
        assert _parse([text[:i], text[i:]]) == [
            (START, ""),
            (THOUGHT, "Hmm."),
            (END, ""),
            (TEXT, "Hi"),
        ], f"split at {i}"


def test_whitespace_before_the_thought_is_allowed():
    assert _parse(["\n", "<think>Hmm.</think>Hi"]) == [
        (START, ""),
        (THOUGHT, "Hmm."),
        (END, ""),
        (TEXT, "Hi"),
    ]


def test_whitespace_after_the_thought_is_dropped_across_chunks():
    assert _parse(["<think>Hmm.</think>", "\n", "\n ", "Hi there", "\n"]) == [
        (START, ""),
        (THOUGHT, "Hmm."),
        (END, ""),
        (TEXT, "Hi there\n"),
    ]


def test_text_without_a_thought_passes_through_unchanged():
    assert _parse([" Hel", "lo <think>", "</think>"]) == [(TEXT, " Hello <think></think>")]


def test_text_that_rules_out_the_tag_is_not_held_back():
    parser = TaggedLLMTextParser("<think>", "</think>")
    assert parser.parse_next_chunk("<th") == []
    assert _pairs(parser.parse_next_chunk("e")) == [(TEXT, "<the")]
    assert _pairs(parser.parse_next_chunk("re")) == [(TEXT, "re")]


def test_partial_tag_is_flushed_at_the_end():
    assert _parse(["<"]) == [(TEXT, "<")]


def test_thought_left_open_is_closed_at_the_end():
    assert _parse(["<think>Hmm", "</thi"]) == [(START, ""), (THOUGHT, "Hmm</thi"), (END, "")]


def test_text_cut_short_is_not_flushed():
    # Inside a thought, the last len("</think>") - 1 characters are held back.
    assert _parse(["<think>Let me check.", "</thi"], flush=False) == [
        (START, ""),
        (THOUGHT, "Let me chec"),
        (END, ""),
    ]
    assert _parse(["<thi"], flush=False) == []


def test_flush_readies_the_parser_for_the_next_text():
    parser = TaggedLLMTextParser("<thinking>", "</thinking>")
    for text in ["<thinking>One.</thinking>", "<thinking>Two.</thinking>"]:
        chunks = parser.parse_next_chunk(text) + parser.flush()
        assert [chunk.kind for chunk in chunks] == [START, THOUGHT, END]
    assert parser.flush() == []
