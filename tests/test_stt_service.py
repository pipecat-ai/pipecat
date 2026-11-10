#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio handling shared by every STTService."""

from collections.abc import AsyncGenerator

import numpy as np
import pytest

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    STTMuteFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# Distinct payloads so the audio handed to run_stt() shows which frame it came from.
INPUT_PCM = b"\x01\x00" * 160
OUTPUT_PCM = b"\x02\x00" * 160


def _make_capturing_service(send_sample_rate: int | None = None, **kwargs) -> STTService:
    """Build an STTService that records the audio handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    """

    class _CapturingSTTService(STTService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.captured: list[bytes] = []

        @property
        def send_sample_rate(self) -> int:
            return send_sample_rate or self.sample_rate

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    return _CapturingSTTService(**kwargs)


def _input_audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=INPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


def _tone(rate: int, seconds: float) -> InputAudioRawFrame:
    t = np.arange(int(rate * seconds)) / rate
    audio = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=rate, num_channels=1)


async def _resampled(frame: InputAudioRawFrame, out_rate: int) -> bytes:
    return await create_stream_resampler().resample(frame.audio, frame.sample_rate, out_rate)


def _output_audio() -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=OUTPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_without_being_transcribed():
    service = _make_capturing_service()

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_even_without_audio_passthrough():
    service = _make_capturing_service(audio_passthrough=False)

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


@pytest.mark.asyncio
async def test_input_audio_at_another_rate_is_resampled_to_the_sample_rate():
    service = _make_capturing_service()
    frame = _tone(24000, 0.1)

    await run_test(
        service,
        frames_to_send=[frame],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame],
    )

    assert service.captured == [await _resampled(frame, SAMPLE_RATE)]


@pytest.mark.asyncio
async def test_input_audio_is_resampled_to_the_send_sample_rate():
    service = _make_capturing_service(send_sample_rate=8000)
    frame = _tone(SAMPLE_RATE, 0.1)

    await run_test(
        service,
        frames_to_send=[frame],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame],
    )

    assert service.captured == [await _resampled(frame, 8000)]


@pytest.mark.asyncio
async def test_usage_is_counted_at_the_send_sample_rate():
    service = _make_capturing_service(send_sample_rate=8000)
    service._sample_rate = 48000

    await service.process_audio_frame(_tone(48000, 1.0), FrameDirection.DOWNSTREAM)

    assert len(service.captured[0]) == 8000 * 2
    assert service._stt_usage_pending_seconds == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_muting_starts_the_resampled_audio_afresh():
    service = _make_capturing_service()
    before, after = _tone(24000, 0.1), _tone(24000, 0.1)

    await run_test(
        service,
        frames_to_send=[before, STTMuteFrame(mute=True), STTMuteFrame(mute=False), after],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, InputAudioRawFrame],
    )

    assert service.captured[-1] == await _resampled(after, SAMPLE_RATE)
