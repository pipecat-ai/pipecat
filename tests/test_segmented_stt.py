#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import wave
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# Distinct, non-zero 16-bit samples so a misread WAV header would be obvious.
PCM = bytes(range(0, 240)) * 4  # 960 bytes, even length


class _CapturingSegmentedSTTService(SegmentedSTTService):
    """Captures the bytes the base class hands to run_stt()."""

    def __init__(self, **kwargs):
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        self.captured: list[bytes] = []

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        self.captured.append(audio)
        return
        yield  # make this an async generator


async def _drive_one_segment(service: SegmentedSTTService):
    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
    )


@pytest.mark.asyncio
async def test_default_mode_wraps_segment_in_wav():
    service = _CapturingSegmentedSTTService()
    assert service.wants_wav_segments is True

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    audio = service.captured[0]

    # A valid WAV container with the right sample rate and the exact PCM payload.
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.asyncio
async def test_passthrough_mode_preserves_exact_pcm():
    class _PCMService(_CapturingSegmentedSTTService):
        @property
        def wants_wav_segments(self) -> bool:
            return False

    service = _PCMService()
    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # Raw PCM, byte-for-byte: no WAV header prepended.
    assert service.captured[0] == PCM
