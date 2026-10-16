#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.utils.sent_audio_timeline import SentAudioTimeline

MS = 1_000_000


def _timeline() -> SentAudioTimeline:
    # Two 20 ms chunks at 16 kHz, the second resuming 980 ms later on the input.
    timeline = SentAudioTimeline()
    timeline.append(100 * MS, 320, 16000)
    timeline.append(1_100 * MS, 320, 16000)
    return timeline


def test_offsets_map_into_the_chunk_that_holds_them():
    timeline = _timeline()
    assert timeline.to_pts(0.0) == 100 * MS
    assert timeline.to_pts(0.015) == 115 * MS
    assert timeline.to_pts(0.025) == 1_105 * MS


def test_a_start_on_a_boundary_takes_the_later_chunk_and_an_end_the_earlier():
    timeline = _timeline()
    assert timeline.to_pts(0.02) == 1_100 * MS
    assert timeline.to_pts(0.02, end=True) == 120 * MS


def test_the_end_of_the_stream_maps_only_as_an_end():
    timeline = _timeline()
    assert timeline.to_pts(0.04, end=True) == 1_120 * MS
    assert timeline.to_pts(0.04) is None
    assert timeline.to_pts(0.0, end=True) is None
    assert timeline.to_pts(0.041, end=True) is None


def test_audio_without_pts_advances_the_stream_but_maps_nowhere():
    timeline = SentAudioTimeline()
    timeline.append(None, 320, 16000)
    timeline.append(500 * MS, 320, 16000)
    assert timeline.to_pts(0.01) is None
    assert timeline.to_pts(0.03) == 510 * MS


def test_chunks_at_different_rates_share_one_stream():
    timeline = SentAudioTimeline()
    timeline.append(0, 480, 24000)
    timeline.append(1_000 * MS, 160, 8000)
    assert timeline.to_pts(0.03) == 1_010 * MS


def test_reset_starts_offsets_from_zero_again():
    timeline = _timeline()
    timeline.reset()
    assert timeline.to_pts(0.0) is None
    timeline.append(9_000 * MS, 320, 16000)
    assert timeline.to_pts(0.005) == 9_005 * MS
