#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WordTimestampTracker."""

from pipecat.utils.context.word_timestamp_tracker import WordTimestamp, WordTimestampTracker

_SECOND = 1_000_000_000


def _word(word: str, timestamp: float) -> WordTimestamp:
    return WordTimestamp(word, timestamp, "ctx", False)


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


def test_reply_last_pts_counts_only_the_replys_own_words():
    """A reply that times no words has no last word, even after an earlier reply's."""
    tracker = WordTimestampTracker()
    tracker.start_reply()
    assert tracker.reply_last_pts is None

    tracker.last_pts = 2 * _SECOND
    assert tracker.reply_last_pts == 2 * _SECOND

    tracker.start_reply()
    assert tracker.reply_last_pts is None
