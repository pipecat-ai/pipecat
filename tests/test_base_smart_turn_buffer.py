#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for BaseSmartTurn's audio buffer behavior.

The buffer holds int16 PCM views and defers the float32 conversion to
_process_speech_segment so it runs once per turn instead of once per
appended audio frame. These tests verify:

  1. The deferred float32 segment is bit-identical to the eager version that
     converted each chunk in append_audio.
  2. The pre-speech trim loop still bounds buffer size when no speech ever
     triggers.
"""

import time
from typing import Any

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")  # noqa: F841 -- needed by BaseSmartTurn subclasses

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState  # noqa: E402
from pipecat.audio.turn.smart_turn.base_smart_turn import (  # noqa: E402
    BaseSmartTurn,
    SmartTurnParams,
)


class _RecordingSmartTurn(BaseSmartTurn):
    """Test double that records the segment_audio passed into _predict_endpoint."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.captured_segment: np.ndarray | None = None

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        self.captured_segment = audio_array.copy()
        return {"prediction": 1, "probability": 0.99}


def _pcm_bytes(values: np.ndarray) -> bytes:
    """Pack a numpy int16 array into PCM bytes, the wire format append_audio expects."""
    return values.astype(np.int16).tobytes()


def test_segment_matches_eager_conversion_bit_identical():
    """The deferred segment must be bit-identical to per-chunk conversion.

    For ``int16 -> float32 / 32768.0`` the two orderings (cast-each-then-concat
    vs concat-then-cast) are equivalent element-wise: the cast is per-element
    and the divide is multiplication by a float constant, so concatenation
    distributes through both. We assert this concretely so that any future
    code change to the conversion pipeline cannot silently shift outputs.
    """
    rng = np.random.default_rng(seed=1234)
    chunk_size = 320  # 20 ms @ 16 kHz
    n_chunks = 12
    chunks_int16 = [
        rng.integers(-32768, 32767, size=chunk_size, dtype=np.int16) for _ in range(n_chunks)
    ]

    # Eager: convert each chunk to float32 in isolation, then concatenate.
    eager_float32_chunks = [c.astype(np.float32) / 32768.0 for c in chunks_int16]
    eager_segment = np.concatenate(eager_float32_chunks)

    # Deferred (what _process_speech_segment now does): concatenate int16, then cast.
    deferred_segment = np.concatenate(chunks_int16).astype(np.float32) / 32768.0

    assert deferred_segment.dtype == np.float32
    assert eager_segment.shape == deferred_segment.shape
    # Bit-identical, not just close: this is a structural property of the
    # operation, not an approximation. Use array_equal to fail loudly if it
    # ever becomes "merely close".
    assert np.array_equal(eager_segment, deferred_segment)


def test_buffer_holds_int16_views_after_append():
    """After append_audio, _audio_buffer entries should be int16, not float32."""
    analyzer = _RecordingSmartTurn(sample_rate=16_000, params=SmartTurnParams())
    analyzer.set_sample_rate(16_000)
    chunk = np.array([100, -200, 300, -400], dtype=np.int16)
    analyzer.append_audio(_pcm_bytes(chunk), is_speech=True)

    assert len(analyzer._audio_buffer) == 1
    _, stored = analyzer._audio_buffer[0]
    assert stored.dtype == np.int16
    np.testing.assert_array_equal(stored, chunk)


def test_segment_audio_is_normalized_float32():
    """When a turn completes, _process_speech_segment must emit float32 in [-1, 1]."""
    analyzer = _RecordingSmartTurn(sample_rate=16_000, params=SmartTurnParams())
    analyzer.set_sample_rate(16_000)

    # Feed a speech-then-silence sequence: a few speech frames, then enough
    # silence to cross stop_secs (default 3 s) so the turn closes.
    chunk_size = 320  # 20 ms @ 16 kHz
    speech = np.array([16_000] * chunk_size, dtype=np.int16)  # constant 0.488...
    silence = np.zeros(chunk_size, dtype=np.int16)

    # Two speech chunks (40 ms) to trigger.
    for _ in range(2):
        analyzer.append_audio(_pcm_bytes(speech), is_speech=True)
    # Then ~3.1 s of silence to exceed stop_secs.
    state = EndOfTurnState.INCOMPLETE
    for _ in range(160):  # 160 * 20 ms = 3.2 s
        state = analyzer.append_audio(_pcm_bytes(silence), is_speech=False)
        if state == EndOfTurnState.COMPLETE:
            break

    # Run the segment processor synchronously (it normally runs on the executor).
    # We rebuild the audio buffer first by re-feeding, because _clear() emptied
    # it when stop_secs fired. Easier path: drive a fresh sequence and call
    # _process_speech_segment directly.
    analyzer2 = _RecordingSmartTurn(sample_rate=16_000, params=SmartTurnParams())
    analyzer2.set_sample_rate(16_000)
    analyzer2._speech_triggered = True
    analyzer2._speech_start_time = time.monotonic()
    for _ in range(4):
        analyzer2.append_audio(_pcm_bytes(speech), is_speech=True)

    state, _metrics = analyzer2._process_speech_segment(analyzer2._audio_buffer)
    assert analyzer2.captured_segment is not None
    assert analyzer2.captured_segment.dtype == np.float32
    assert float(np.max(np.abs(analyzer2.captured_segment))) <= 1.0
    # The constant 16000 / 32768 ≈ 0.48828125 should appear throughout.
    assert np.allclose(analyzer2.captured_segment, 16_000 / 32768.0)


def test_pre_speech_buffer_trim_still_bounds_growth():
    """Without speech, the buffer must stay bounded by pre_speech_ms + stop_secs + max_duration_secs."""
    params = SmartTurnParams(pre_speech_ms=100, stop_secs=0.2, max_duration_secs=0.5)
    analyzer = _RecordingSmartTurn(sample_rate=16_000, params=params)
    analyzer.set_sample_rate(16_000)

    chunk = np.array([0] * 320, dtype=np.int16)
    # Feed 200 chunks of non-speech (4 seconds wall time would be too slow; just
    # rely on the fact that all timestamps are taken via time.monotonic() inside
    # append_audio, and any chunk older than 0.8 s gets popped on each call).
    for _ in range(50):
        analyzer.append_audio(_pcm_bytes(chunk), is_speech=False)
        time.sleep(0.02)  # advance monotonic clock past the trim window cumulatively

    # max_buffer_time is 0.1 + 0.2 + 0.5 = 0.8 s; at 50 Hz that caps the buffer
    # well below 50 entries. We're conservative — anything < 50 proves the
    # trim ran.
    assert len(analyzer._audio_buffer) < 50


def test_clear_prevents_stale_stop_secs_completion():
    """clear() after an externally-ended turn must kill the stale silence timer.

    Regression for the phantom end-of-turn: a turn force-ended mid-speech
    (e.g. a max-speech cap plus a mute strategy) leaves _speech_triggered
    True with buffered audio. Without clear(), post-unmute silence
    accumulates until stop_secs fires a spurious COMPLETE during the bot's
    speech.
    """
    params = SmartTurnParams(stop_secs=0.2)
    analyzer = _RecordingSmartTurn(sample_rate=16_000, params=params)
    analyzer.set_sample_rate(16_000)

    chunk_size = 320  # 20 ms @ 16 kHz
    speech = np.array([16_000] * chunk_size, dtype=np.int16)
    silence = np.zeros(chunk_size, dtype=np.int16)

    # Mid-speech state: speech triggered, no COMPLETE yet.
    for _ in range(4):
        analyzer.append_audio(_pcm_bytes(speech), is_speech=True)
    assert analyzer.speech_triggered

    # The turn is ended externally: the strategy's handle_user_turn_stopped
    # callback calls clear().
    analyzer.clear()
    assert not analyzer.speech_triggered

    # Ambient silence well past stop_secs (0.6 s vs 0.2 s). Without the
    # clear, this returned COMPLETE: the phantom stop.
    for _ in range(30):
        state = analyzer.append_audio(_pcm_bytes(silence), is_speech=False)
        assert state == EndOfTurnState.INCOMPLETE


def test_clear_drops_pre_clear_audio_from_next_segment():
    """After clear(), the next prediction must only see post-clear audio."""
    analyzer = _RecordingSmartTurn(sample_rate=16_000, params=SmartTurnParams())
    analyzer.set_sample_rate(16_000)

    chunk_size = 320
    old = np.full(chunk_size, 1_000, dtype=np.int16)
    new = np.full(chunk_size, 2_000, dtype=np.int16)

    # Audio from the force-ended turn.
    for _ in range(4):
        analyzer.append_audio(_pcm_bytes(old), is_speech=True)
    analyzer.clear()

    # The next turn's audio, fed continuously as usual.
    for _ in range(4):
        analyzer.append_audio(_pcm_bytes(new), is_speech=True)

    analyzer._process_speech_segment(analyzer._audio_buffer)
    assert analyzer.captured_segment is not None
    # Only the post-clear constant may appear in the segment.
    assert np.allclose(analyzer.captured_segment, 2_000 / 32768.0)
