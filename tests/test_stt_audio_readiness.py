#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    STTMetadataFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# 100ms of 16-bit mono audio per frame.
FRAME_BYTES = SAMPLE_RATE // 10 * 2


def _audio(marker: int) -> InputAudioRawFrame:
    """Build an audio frame whose payload identifies it."""
    return InputAudioRawFrame(
        audio=bytes([marker]) * FRAME_BYTES, sample_rate=SAMPLE_RATE, num_channels=1
    )


def _make_capturing_service(**kwargs) -> STTService:
    """Build an STTService that records the audio handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    """

    class _CapturingSTTService(STTService):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.received: list[int] = []

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
            self.received.append(audio[0])
            yield None

    service = _CapturingSTTService(**kwargs)
    # Audio only reaches a service after StartFrame, which is what sets this.
    service._sample_rate = SAMPLE_RATE
    return service


@pytest.mark.asyncio
async def test_audio_is_held_until_ready_then_replayed_in_order():
    service = _make_capturing_service()
    service._clear_audio_ready()

    for marker in (1, 2, 3):
        await service.process_audio_frame(_audio(marker), FrameDirection.DOWNSTREAM)

    assert service.received == []

    # Readiness alone does not replay: held audio drains on the next audio
    # frame, so replay stays on the audio path and cannot interleave.
    service._set_audio_ready()
    assert service.received == []

    await service.process_audio_frame(_audio(4), FrameDirection.DOWNSTREAM)
    assert service.received == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_held_audio_is_bounded_and_keeps_the_most_recent():
    # Room for two 100ms frames.
    service = _make_capturing_service(max_pending_audio_seconds=0.2)
    service._clear_audio_ready()

    for marker in (1, 2, 3, 4):
        await service.process_audio_frame(_audio(marker), FrameDirection.DOWNSTREAM)

    service._set_audio_ready()
    await service.process_audio_frame(_audio(5), FrameDirection.DOWNSTREAM)

    # The oldest audio is discarded, so the most recent speech survives.
    assert service.received == [3, 4, 5]


@pytest.mark.asyncio
async def test_bound_still_applies_after_an_earlier_hold_drained():
    """A second hold, e.g. across a reconnect, is bounded like the first."""
    service = _make_capturing_service(max_pending_audio_seconds=0.2)

    service._clear_audio_ready()
    await service.process_audio_frame(_audio(1), FrameDirection.DOWNSTREAM)
    service._set_audio_ready()
    await service.process_audio_frame(_audio(2), FrameDirection.DOWNSTREAM)
    assert service.received == [1, 2]

    service._clear_audio_ready()
    for marker in (3, 4, 5, 6):
        await service.process_audio_frame(_audio(marker), FrameDirection.DOWNSTREAM)
    service._set_audio_ready()
    await service.process_audio_frame(_audio(7), FrameDirection.DOWNSTREAM)

    assert service.received == [1, 2, 5, 6, 7]


@pytest.mark.asyncio
async def test_interruption_discards_held_audio(monkeypatch):
    # Only STTService's own frame handling is under test here; the base
    # implementation needs a live pipeline to start an interruption.
    monkeypatch.setattr(AIService, "process_frame", AsyncMock())

    service = _make_capturing_service()
    service._clear_audio_ready()

    await service.process_audio_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    service._set_audio_ready()
    await service.process_audio_frame(_audio(2), FrameDirection.DOWNSTREAM)

    # Audio from before the interruption is no longer worth transcribing.
    assert service.received == [2]


@pytest.mark.asyncio
async def test_muted_audio_is_dropped_on_replay_not_when_held():
    service = _make_capturing_service()
    service._clear_audio_ready()

    await service.process_audio_frame(_audio(1), FrameDirection.DOWNSTREAM)
    service._muted = True

    service._set_audio_ready()
    await service.process_audio_frame(_audio(2), FrameDirection.DOWNSTREAM)

    # Mute is applied when the audio is actually sent, so a service muted while
    # audio was held does not transcribe it.
    assert service.received == []


@pytest.mark.asyncio
async def test_service_is_ready_by_default():
    """A service that never touches readiness passes audio straight through."""
    service = _make_capturing_service()

    await run_test(
        service,
        frames_to_send=[_audio(1), _audio(2)],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, InputAudioRawFrame],
    )

    assert service.received == [1, 2]
