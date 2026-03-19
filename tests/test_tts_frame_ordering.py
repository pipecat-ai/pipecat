#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame ordering across TTS service types.

Covers three patterns:
- HTTP TTS services (e.g. CartesiaHttpTTSService): yield audio frames synchronously.
- WebSocket TTS services without pause (e.g. CartesiaTTSService): deliver audio via
  append_to_audio_context from a background receive loop, no frame-processing pause.
- WebSocket TTS services with pause (e.g. ElevenLabsTTSService): same delivery
  mechanism, but pause downstream frame processing while audio is in flight.

For all three patterns we verify:
    AggregatedTextFrame → TTSStartedFrame → TTSAudioRawFrame (1+) → TTSStoppedFrame → FooFrame

repeated for each TTSSpeakFrame, with no cross-group contamination.
"""

import asyncio
import unittest
from dataclasses import dataclass
from typing import AsyncGenerator, List, Sequence, Tuple

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    DataFrame,
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test

# ---------------------------------------------------------------------------
# Test-only frame
# ---------------------------------------------------------------------------

_FAKE_AUDIO = b"\x00\x01" * 320  # 320 bytes of silence
_SAMPLE_RATE = 16000


@dataclass
class FooFrame(DataFrame):
    """Marker frame used to verify relative ordering against TTS audio frames."""

    label: str = ""


# ---------------------------------------------------------------------------
# Mock TTS services
# ---------------------------------------------------------------------------


class MockHttpTTSService(TTSService):
    """Simulates an HTTP TTS service (e.g. CartesiaHttpTTSService).

    Audio frames are yielded synchronously from run_tts(), so the audio context
    is fully populated before the next downstream frame is processed.
    TTSStoppedFrame is appended by the base class in on_turn_context_completed()
    once it detects _is_yielding_frames_synchronously is True.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class MockWebSocketTTSService(TTSService):
    """Simulates a WebSocket TTS service without frame-processing pause (e.g. CartesiaTTSService).

    run_tts() is an empty async generator (signals async delivery). A background
    task appends audio frames and the TTSStoppedFrame to the audio context after a
    short delay, mimicking real WebSocket receive-loop behaviour.
    pause_frame_processing=False means downstream frames (FooFrame) are NOT held.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver_audio():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver_audio(), name=f"mock_ws_deliver_{context_id}")
        if False:
            yield  # make this an async generator without yielding anything


class MockWebSocketPauseTTSService(TTSService):
    """Simulates a WebSocket TTS service WITH frame-processing pause (e.g. ElevenLabsTTSService).

    Identical to MockWebSocketTTSService except pause_frame_processing=True.
    on_audio_context_completed() resumes downstream processing once the full
    audio context has been pushed, guaranteeing FooFrame arrives after TTSStoppedFrame.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def on_audio_context_completed(self, context_id: str):
        # Resume frame processing after the audio context is fully played out.
        await self._maybe_resume_frame_processing()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver_audio():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver_audio(), name=f"mock_ws_pause_deliver_{context_id}")
        if False:
            yield


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------


def _assert_group_ordering(
    down_frames: Sequence[Frame],
    expected_groups: List[Tuple[str, str]],
) -> None:
    """Assert two (or more) TTS+FooFrame groups are in strict order.

    Args:
        down_frames: All downstream frames received by the test sink.
        expected_groups: List of (tts_text, foo_label) pairs, one per TTSSpeakFrame.
            tts_text is unused in assertions today but included for readability.
    """
    relevant = [
        f
        for f in down_frames
        if isinstance(
            f, (AggregatedTextFrame, TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame, FooFrame)
        )
    ]

    # Locate the FooFrames that delimit groups.
    foo_indices = [i for i, f in enumerate(relevant) if isinstance(f, FooFrame)]
    assert len(foo_indices) == len(expected_groups), (
        f"Expected {len(expected_groups)} FooFrames, got {len(foo_indices)}.\n"
        f"Relevant frames: {[type(f).__name__ for f in relevant]}"
    )

    # Build groups: everything up to and including each FooFrame.
    groups: List[List[Frame]] = []
    prev = 0
    for idx in foo_indices:
        groups.append(relevant[prev : idx + 1])
        prev = idx + 1

    for group, (_, foo_label) in zip(groups, expected_groups):
        types = [type(f) for f in group]
        type_names = [t.__name__ for t in types]

        assert AggregatedTextFrame in types, (
            f"Group {foo_label!r}: missing AggregatedTextFrame. Got: {type_names}"
        )
        assert TTSStartedFrame in types, (
            f"Group {foo_label!r}: missing TTSStartedFrame. Got: {type_names}"
        )
        assert TTSAudioRawFrame in types, (
            f"Group {foo_label!r}: missing TTSAudioRawFrame. Got: {type_names}"
        )
        assert TTSStoppedFrame in types, (
            f"Group {foo_label!r}: missing TTSStoppedFrame. Got: {type_names}"
        )

        started_idx = types.index(TTSStartedFrame)
        stopped_idx = types.index(TTSStoppedFrame)
        foo_idx = types.index(FooFrame)

        assert started_idx < stopped_idx, (
            f"Group {foo_label!r}: TTSStartedFrame (pos {started_idx}) must precede "
            f"TTSStoppedFrame (pos {stopped_idx}). Got: {type_names}"
        )
        assert stopped_idx < foo_idx, (
            f"Group {foo_label!r}: TTSStoppedFrame (pos {stopped_idx}) must precede "
            f"FooFrame (pos {foo_idx}). Got: {type_names}"
        )

        # All frames between TTSStartedFrame and TTSStoppedFrame must be audio.
        mid_types = types[started_idx + 1 : stopped_idx]
        for t in mid_types:
            assert t is TTSAudioRawFrame, (
                f"Group {foo_label!r}: unexpected frame {t.__name__!r} between "
                f"TTSStartedFrame and TTSStoppedFrame. Got: {type_names}"
            )

        # Check the FooFrame label.
        actual_label = group[foo_idx].label
        assert actual_label == foo_label, (
            f"Expected FooFrame(label={foo_label!r}), got label={actual_label!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_GROUPS = [("test 1", "1"), ("test 2", "2")]


def _make_frames_no_sleep() -> List[Frame]:
    """Return two TTSSpeakFrame+FooFrame pairs sent back-to-back.

    Only correct for services that pause downstream processing until the audio
    context is fully consumed (pause_frame_processing=True + on_audio_context_completed).
    """
    return [
        TTSSpeakFrame(text="test 1", append_to_context=False),
        FooFrame(label="1"),
        TTSSpeakFrame(text="test 2", append_to_context=False),
        FooFrame(label="2"),
    ]


def _print_frames_received(frames_received) -> None:
    print("FRAMES RECEIVED:")
    for frame in frames_received[0]:
        print(frame.name)


@pytest.mark.asyncio
async def test_http_tts_frame_ordering():
    """HTTP TTS services yield audio synchronously."""
    tts = MockHttpTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())

    # only for debugging
    _print_frames_received(frames_received)

    _assert_group_ordering(frames_received[0], _GROUPS)


@pytest.mark.asyncio
async def test_websocket_tts_no_pause_frame_ordering():
    """WebSocket TTS services without pause_frame_processing."""
    tts = MockWebSocketTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())
    _assert_group_ordering(frames_received[0], _GROUPS)


@pytest.mark.asyncio
async def test_websocket_tts_with_pause_frame_ordering():
    """WebSocket TTS services with pause_frame_processing=True."""
    tts = MockWebSocketPauseTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())
    _assert_group_ordering(frames_received[0], _GROUPS)


if __name__ == "__main__":
    unittest.main()
