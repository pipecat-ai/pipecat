#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WordTimestampTracker."""

from pipecat.utils.context.word_timestamp_tracker import CachedWord, WordTimestampTracker

_SECOND = 1_000_000_000


def _word(word: str, timestamp: float) -> CachedWord:
    return CachedWord(word, timestamp, "ctx", False)


def test_words_are_timed_from_when_timing_starts():
    """A word's PTS is the start time plus its time in the audio."""
    tracker = WordTimestampTracker()
    assert not tracker.started
    tracker.start(10 * _SECOND)

    assert tracker.started
    assert tracker.pts(0.5) == 10 * _SECOND + _SECOND // 2


def test_timing_starts_after_the_last_word_if_it_is_later():
    """Words of the next context come after the last word still to play."""
    tracker = WordTimestampTracker()
    tracker.last_pts = 12 * _SECOND
    tracker.start(10 * _SECOND)

    assert tracker.pts(0) == 12 * _SECOND


def test_start_hands_back_cached_words_once():
    """Words reported before timing starts come back in order, only the first time."""
    tracker = WordTimestampTracker()
    first, second = _word("Hello", 0.0), _word("there", 0.4)
    tracker.cache(first)
    tracker.cache(second)

    assert tracker.start(10 * _SECOND) == [first, second]
    assert tracker.start(20 * _SECOND) == []
    assert tracker.pts(0) == 10 * _SECOND


def test_reset_stops_timing_and_drops_cached_words():
    """After a reset, timing starts again and earlier cached words are gone."""
    tracker = WordTimestampTracker()
    tracker.start(10 * _SECOND)
    tracker.cache(_word("stale", 0.0))
    tracker.reset()

    assert not tracker.started
    assert tracker.start(20 * _SECOND) == []
    assert tracker.pts(0) == 20 * _SECOND


def test_shift_moves_timed_and_coming_words_later():
    """Words timed so far and the ones still to come move by the same offset."""
    tracker = WordTimestampTracker()
    tracker.start(10 * _SECOND)
    tracker.last_pts = tracker.pts(0.5)
    tracker.shift(2 * _SECOND)

    assert tracker.baseline == 12 * _SECOND
    assert tracker.last_pts == 12 * _SECOND + _SECOND // 2
    assert tracker.pts(1.0) == 13 * _SECOND


def test_shift_keeps_an_untimed_last_word_untimed():
    """Zero means no word has been timed yet, so it stays zero."""
    tracker = WordTimestampTracker()
    tracker.shift(2 * _SECOND)

    assert tracker.last_pts == 0
    assert tracker.uninterruptible_last_pts == 0
    assert not tracker.started
