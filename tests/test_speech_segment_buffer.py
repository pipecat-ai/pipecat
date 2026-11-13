#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the buffer that keeps one stretch of speech with the audio before it."""

from pipecat.audio.speech_segment_buffer import SpeechSegmentBuffer

#: A sample rate where one second of 16-bit mono audio is 20 bytes.
SAMPLE_RATE = 10


def _seconds(value: int, secs: float) -> bytes:
    """``secs`` seconds of audio whose every byte is ``value``."""
    return bytes([value]) * int(SAMPLE_RATE * secs * 2)


def _buffer(**kwargs) -> SpeechSegmentBuffer:
    buffer = SpeechSegmentBuffer(**kwargs)
    buffer.set_sample_rate(SAMPLE_RATE)
    return buffer


def test_a_segment_starts_with_the_last_pre_speech_secs_before_it():
    buffer = _buffer(pre_speech_secs=1.0)

    buffer.append(_seconds(1, 2))
    buffer.append(_seconds(2, 0.5))
    buffer.start()
    buffer.append(_seconds(3, 3))

    assert buffer.stop() == _seconds(1, 0.5) + _seconds(2, 0.5) + _seconds(3, 3)


def test_stopping_empties_the_buffer_for_the_next_segment():
    buffer = _buffer()

    buffer.start()
    buffer.append(_seconds(1, 1))
    buffer.stop()
    buffer.append(_seconds(2, 0.5))
    buffer.start()
    buffer.append(_seconds(3, 1))

    assert buffer.stop() == _seconds(2, 0.5) + _seconds(3, 1)


def test_a_segment_keeps_its_last_max_secs():
    buffer = _buffer(pre_speech_secs=1.0, max_secs=2.0)

    buffer.append(_seconds(1, 1))
    buffer.start()
    buffer.append(_seconds(2, 3))

    assert buffer.stop() == _seconds(2, 2)


def test_a_segment_without_max_secs_keeps_everything():
    buffer = _buffer(pre_speech_secs=0.5)

    buffer.append(_seconds(1, 1))
    buffer.start()
    buffer.append(_seconds(2, 60))

    assert buffer.stop() == _seconds(1, 0.5) + _seconds(2, 60)
