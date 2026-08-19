#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared XML/SSML markup utilities for text matching and cursor advancement.

Two notions of "what is markup" live here, and they differ on a single point --
a ``'<'`` with no later ``'>'``:

- :func:`strip_markup` treats it as an open tag and swallows the rest of the
  string. Correct for a *fragment* that may have been cut mid-tag, such as a
  single token from a TTS word-timestamp stream.
- :func:`strip_complete_markup` treats it as content. Correct for a *complete*
  text, where a lone ``'<'`` is real (``"5 < 10"``, ``"<3"``).

Every markup decision in the word-timestamp path routes through one of the two,
so the callers can't disagree about which characters are a tag.
"""

import re
from collections.abc import Iterator


def _iter_clean_chars(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(raw_index, char)`` for each character of *text* outside markup.

    The fragment-tolerant definition of markup -- anything between '<' and '>',
    syntax-based and tag-name independent -- shared by :func:`strip_markup` and
    :func:`raw_offset_after_clean_chars` so the two can't disagree. An unclosed '<'
    swallows the rest of the string.
    """
    in_tag = False
    for i, ch in enumerate(text):
        if in_tag:
            if ch == ">":
                in_tag = False
        elif ch == "<":
            in_tag = True
        else:
            yield i, ch


def strip_markup(text: str) -> str:
    """Remove XML/SSML-like markup from a possibly-truncated text fragment.

    Syntax-based, not tag-name based: treats anything between '<' and '>' as
    markup and preserves text outside it. An unclosed '<' swallows the rest of
    *text*, matching how a raw word-timestamp token can arrive mid-tag (see
    :func:`_iter_clean_chars`).

    For a *complete* text, use :func:`strip_complete_markup` instead.

    Used by :class:`~pipecat.utils.context.text_segment_map.TextSegmentMap` to
    match a word against a segment, where the incoming word may be a fragment of
    a still-open tag.
    """
    return "".join(ch for _, ch in _iter_clean_chars(text))


_COMPLETE_MARKUP_RE = re.compile(r"<[^>]+>")
"""Matched '<...>' pairs in a complete text.

The definition of markup for a static text, shared by
:func:`strip_complete_markup` and :func:`split_markup_runs` so the two can't
disagree about which characters a segment split may treat as a tag.
"""


def strip_complete_markup(text: str) -> str:
    """Remove well-formed '<...>' markup from a complete, static text.

    Unlike :func:`strip_markup`, only strips matched '<...>' pairs -- a lone
    '<' with no later '>' is left in place as real content rather than
    swallowing the rest of *text*, since there is no streamed fragment here
    that could be mid-tag.

    Used by
    :attr:`~pipecat.utils.context.text_segment_map.TextSegment.is_transformed`,
    by :func:`~pipecat.utils.text.alnum_utils.alnum_only`, and by
    :class:`~pipecat.utils.context.word_completion_tracker.WordCompletionTracker`
    to default ``user_facing_text`` to a tag-free string.
    """
    return _COMPLETE_MARKUP_RE.sub("", text)


def raw_offset_after_clean_chars(text: str, n: int) -> int:
    """Return the raw offset into *text* just past its *n*-th markup-stripped char.

    Inverse of :func:`strip_markup` for a prefix: where ``strip_markup`` collects
    every non-markup char, this finds the raw index one past the *n*-th of them --
    converting a match measured in markup-stripped space back to a raw offset.
    Returns ``len(text)`` when *text* has fewer than *n* non-markup chars.
    """
    if n <= 0:
        return 0
    seen = 0
    for i, _ in _iter_clean_chars(text):
        seen += 1
        if seen == n:
            return i + 1
    return len(text)


def split_markup_runs(text: str) -> list[str]:
    """Split *text* into alternating runs of tagged and untagged words.

    A word is considered tagged if it overlaps a complete ``'<...>'`` pair. A lone
    ``'<'`` is treated as content, not as the start of a tag (see
    :func:`strip_complete_markup`). Consecutive words with the same classification
    form a single run, so whitespace inside a tag such as
    ``<phoneme alphabet="ipa">`` never splits the words it spans across runs.

    Example::

        split_markup_runs("I love to count <spell>1234</spell>.")
        # -> ["I love to count ", "<spell>1234</spell>."]

    Text with no markup yields a single run, unchanged.

    Used by :class:`~pipecat.utils.context.text_segment_map.TextSegmentMap` when
    it builds its segments, to give a tag one of its own.
    """
    tag_spans = [m.span() for m in _COMPLETE_MARKUP_RE.finditer(text)]
    if not tag_spans:
        return [text] if text else []

    runs: list[str] = []
    run_is_tagged: bool | None = None
    pos = 0

    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        start, end = pos, pos + len(token)
        pos = end
        is_tagged = any(tag_start < end and start < tag_end for tag_start, tag_end in tag_spans)
        if is_tagged == run_is_tagged:
            runs[-1] += token
        else:
            runs.append(token)
            run_is_tagged = is_tagged

    return runs
