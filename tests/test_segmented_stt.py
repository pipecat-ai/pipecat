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
    MetricsFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# Distinct, non-zero 16-bit samples so a misread WAV header would be obvious.
PCM = bytes(range(0, 240)) * 4  # 960 bytes, even length
# What the default trailing_silence_secs (0.5 s) appends, as 16-bit mono PCM.
DEFAULT_SILENCE = bytes(int(SAMPLE_RATE * 0.5) * 2)


def _make_capturing_service(wants_wav: bool | None = None, **kwargs) -> SegmentedSTTService:
    """Build a SegmentedSTTService that captures the bytes handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py, which
    would try to construct it and fail on its (intentionally minimal) settings.

    Args:
        wants_wav: If None, inherit the base default; otherwise force the
            ``wants_wav_segments`` contract to this value.
        **kwargs: Passed to the service constructor.
    """

    class _CapturingSegmentedSTTService(SegmentedSTTService):
        def __init__(self, **kwargs):
            super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
            self.captured: list[bytes] = []

        def can_generate_metrics(self) -> bool:
            return True

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    if wants_wav is not None:
        _CapturingSegmentedSTTService.wants_wav_segments = property(lambda self: wants_wav)

    return _CapturingSegmentedSTTService(**kwargs)


def _segment_frames() -> list[Frame]:
    return [
        VADUserStartedSpeakingFrame(),
        InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
        VADUserStoppedSpeakingFrame(),
    ]


async def _drive_one_segment(service: SegmentedSTTService):
    await run_test(service, frames_to_send=_segment_frames())


async def _drive_one_segment_with_usage(service: SegmentedSTTService) -> float:
    """Drive one segment with usage metrics on and return the reported audio seconds."""
    received_down, _ = await run_test(
        service,
        frames_to_send=_segment_frames(),
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )
    usage_data = [
        d
        for f in received_down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, STTUsageMetricsData)
    ]
    assert len(usage_data) == 1
    return usage_data[0].value.audio_seconds


def _wav_frames(audio: bytes) -> bytes:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        return wav.readframes(wav.getnframes())


@pytest.mark.asyncio
async def test_default_mode_wraps_segment_in_wav():
    service = _make_capturing_service(trailing_silence_secs=0)
    assert service.wants_wav_segments is True

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # A valid WAV container with the right sample rate and the exact PCM payload.
    assert _wav_frames(service.captured[0]) == PCM


@pytest.mark.asyncio
async def test_passthrough_mode_preserves_exact_pcm():
    service = _make_capturing_service(wants_wav=False, trailing_silence_secs=0)
    assert service.wants_wav_segments is False

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # Raw PCM, byte-for-byte: no WAV header prepended.
    assert service.captured[0] == PCM


@pytest.mark.asyncio
async def test_segment_emits_usage_for_raw_buffer_duration():
    # WAV mode: usage must measure the PCM audio, not the WAV container.
    service = _make_capturing_service(trailing_silence_secs=0)

    audio_seconds = await _drive_one_segment_with_usage(service)

    assert audio_seconds == pytest.approx(len(PCM) / (SAMPLE_RATE * 2))


@pytest.mark.asyncio
async def test_segment_is_padded_with_trailing_silence_by_default():
    service = _make_capturing_service(wants_wav=False)

    await _drive_one_segment(service)

    assert service.captured == [PCM + DEFAULT_SILENCE]


@pytest.mark.asyncio
async def test_wav_segment_includes_trailing_silence():
    service = _make_capturing_service()

    await _drive_one_segment(service)

    assert _wav_frames(service.captured[0]) == PCM + DEFAULT_SILENCE


@pytest.mark.asyncio
async def test_trailing_silence_is_configurable():
    service = _make_capturing_service(wants_wav=False, trailing_silence_secs=0.1)

    await _drive_one_segment(service)

    assert service.captured == [PCM + bytes(int(SAMPLE_RATE * 0.1) * 2)]


@pytest.mark.asyncio
async def test_usage_includes_trailing_silence():
    # The padding is submitted to the provider, so it counts as audio usage.
    service = _make_capturing_service()

    audio_seconds = await _drive_one_segment_with_usage(service)

    assert audio_seconds == pytest.approx((len(PCM) + len(DEFAULT_SILENCE)) / (SAMPLE_RATE * 2))
