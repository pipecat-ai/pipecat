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


def _make_capturing_service(wants_wav: bool | None = None) -> SegmentedSTTService:
    """Build a SegmentedSTTService that captures the bytes handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py, which
    would try to construct it and fail on its (intentionally minimal) settings.

    Args:
        wants_wav: If None, inherit the base default; otherwise force the
            ``wants_wav_segments`` contract to this value.
    """

    class _CapturingSegmentedSTTService(SegmentedSTTService):
        def __init__(self, **kwargs):
            super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
            self.captured: list[bytes] = []

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    if wants_wav is not None:
        _CapturingSegmentedSTTService.wants_wav_segments = property(lambda self: wants_wav)

    return _CapturingSegmentedSTTService()


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
    service = _make_capturing_service()
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
    service = _make_capturing_service(wants_wav=False)
    assert service.wants_wav_segments is False

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # Raw PCM, byte-for-byte: no WAV header prepended.
    assert service.captured[0] == PCM
