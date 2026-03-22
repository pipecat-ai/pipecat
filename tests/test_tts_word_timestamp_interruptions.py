#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class MockWebsocketWordTimestampTTSService(TTSService):
    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._first_word_emitted = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if text == "interrupt-before-audio":
            self.create_task(
                self._deliver_cached_word_before_audio(context_id),
                name=f"cached_word_{context_id}",
            )
        elif text == "interrupt-late-word":
            self.create_task(
                self._deliver_late_word_after_interruption(context_id),
                name=f"late_word_{context_id}",
            )
        elif text == "follow-up":
            self.create_task(
                self._deliver_follow_up_context(context_id),
                name=f"follow_up_{context_id}",
            )
        elif text == "monotonic-guard":
            self.create_task(
                self._deliver_non_monotonic_words(context_id),
                name=f"monotonic_{context_id}",
            )

        if False:
            yield

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and frame.text == "first":
            self._first_word_emitted.set()

    async def _deliver_cached_word_before_audio(self, context_id: str):
        await asyncio.sleep(0.01)
        await self.add_word_timestamps([("stale", 0.09288)], context_id)

    async def _deliver_late_word_after_interruption(self, context_id: str):
        await asyncio.sleep(0.06)
        await self.add_word_timestamps([("late", 0.09288)], context_id)

    async def _deliver_follow_up_context(self, context_id: str):
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
        await self.add_word_timestamps([("fresh", 0.1)], context_id)
        await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)], context_id)
        await self.remove_audio_context(context_id)

    async def _deliver_non_monotonic_words(self, context_id: str):
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
        await self.add_word_timestamps([("first", 0.2)], context_id)
        await asyncio.wait_for(self._first_word_emitted.wait(), timeout=1.0)
        self._initial_word_timestamp = 0
        await self.add_word_timestamps([("second", 0.1)], context_id)
        await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)], context_id)
        await self.remove_audio_context(context_id)


def _text_frames(frames: list[Frame]) -> list[TTSTextFrame]:
    return [frame for frame in frames if isinstance(frame, TTSTextFrame)]


@pytest.mark.asyncio
async def test_interruptions_clear_cached_word_timestamps_before_first_audio():
    tts = MockWebsocketWordTimestampTTSService()

    down_frames, _ = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="interrupt-before-audio", append_to_context=False),
            SleepFrame(0.05),
            InterruptionFrame(),
            TTSSpeakFrame(text="follow-up", append_to_context=False),
            SleepFrame(0.1),
        ],
    )

    assert [frame.text for frame in _text_frames(down_frames)] == ["fresh"]


@pytest.mark.asyncio
async def test_interruptions_drop_late_word_timestamps_for_stale_contexts():
    tts = MockWebsocketWordTimestampTTSService()

    down_frames, _ = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="interrupt-late-word", append_to_context=False),
            SleepFrame(0.02),
            InterruptionFrame(),
            TTSSpeakFrame(text="follow-up", append_to_context=False),
            SleepFrame(0.12),
        ],
    )

    assert [frame.text for frame in _text_frames(down_frames)] == ["fresh"]


@pytest.mark.asyncio
async def test_word_timestamps_are_clamped_to_monotonic_pts():
    tts = MockWebsocketWordTimestampTTSService()

    down_frames, _ = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="monotonic-guard", append_to_context=False),
            SleepFrame(0.15),
        ],
    )

    text_frames = _text_frames(down_frames)

    assert [frame.text for frame in text_frames] == ["first", "second"]
    assert text_frames[1].pts > text_frames[0].pts
