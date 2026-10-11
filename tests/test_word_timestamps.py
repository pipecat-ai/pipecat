#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WordTimestamps."""

from pipecat.utils.context.word_timestamps import CachedWord, WordTimestamps

_SECOND = 1_000_000_000


def _word(word: str, timestamp: float) -> CachedWord:
    return CachedWord(word, timestamp, "ctx", False)


def test_words_are_timed_from_when_timing_starts():
    """A word's PTS is the start time plus its time in the audio."""
    timestamps = WordTimestamps()
    assert not timestamps.started
    timestamps.start(10 * _SECOND)

    assert timestamps.started
    assert timestamps.pts(0.5) == 10 * _SECOND + _SECOND // 2


def test_timing_starts_after_the_last_word_if_it_is_later():
    """Words of the next context come after the last word still to play."""
    timestamps = WordTimestamps()
    timestamps.last_pts = 12 * _SECOND
    timestamps.start(10 * _SECOND)

    assert timestamps.pts(0) == 12 * _SECOND


def test_start_hands_back_cached_words_once():
    """Words reported before timing starts come back in order, only the first time."""
    timestamps = WordTimestamps()
    first, second = _word("Hello", 0.0), _word("there", 0.4)
    timestamps.cache(first)
    timestamps.cache(second)

    assert timestamps.start(10 * _SECOND) == [first, second]
    assert timestamps.start(20 * _SECOND) == []
    assert timestamps.pts(0) == 10 * _SECOND


def test_reset_stops_timing_and_drops_cached_words():
    """After a reset, timing starts again and earlier cached words are gone."""
    timestamps = WordTimestamps()
    timestamps.start(10 * _SECOND)
    timestamps.cache(_word("stale", 0.0))
    timestamps.reset()

    assert not timestamps.started
    assert timestamps.start(20 * _SECOND) == []
    assert timestamps.pts(0) == 20 * _SECOND
