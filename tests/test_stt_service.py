#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio handling shared by every STTService."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

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
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import run_test
from pipecat.utils.types import NOT_GIVEN, NotGiven

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


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ([("Hi", 0), ("there", 0), ("hello", 1)], [(0, "Hi there"), (1, "hello")]),
        ([("Hi", 0), ("there", None)], []),
        ([("Hi", 0), ("missing", 1)], []),
        ([("Hi", 0), ("", 1)], []),
        ([], []),
    ],
)
def test_speaker_runs_from_word_pairs(words, expected):
    """Speaker runs are built from plain (text, speaker) pairs."""
    assert STTService._speaker_runs(" Hi there hello ", words) == expected


def test_speaker_runs_locate_repeated_words_in_order():
    """A repeated word is matched after the previous one, not at its first occurrence."""
    runs = STTService._speaker_runs(
        "no, no, no.",
        [("no", 0), ("no", 1), ("no", 0)],
    )

    assert runs == [(0, "no,"), (1, "no,"), (0, "no.")]


def test_speaker_runs_keep_provider_label_type():
    """String labels pass through unchanged."""
    runs = STTService._speaker_runs("Hi. Hello.", [("Hi.", "A"), ("Hello.", "B")])

    assert runs == [("A", "Hi."), ("B", "Hello.")]


@pytest.mark.parametrize(
    ("speaker_format", "speaker_label", "expected"),
    [
        ("{speaker}: {text}", "A", "A: hello"),
        ("{speaker}: {text}", None, "hello"),
        (None, "A", "hello"),
    ],
)
def test_apply_speaker_format(speaker_format, speaker_label, expected):
    """The format is applied only when both a format and a label are present."""

    @dataclass
    class _DiarizingSettings(STTSettings):
        speaker_format: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    service = _make_capturing_service(
        settings=_DiarizingSettings(model=None, language=None, speaker_format=speaker_format)
    )

    assert service._apply_speaker_format("hello", speaker_label) == expected


def test_apply_speaker_format_without_the_setting():
    """A service whose settings have no ``speaker_format`` leaves text unchanged."""
    service = _make_capturing_service()

    assert service._apply_speaker_format("hello", "A") == "hello"
