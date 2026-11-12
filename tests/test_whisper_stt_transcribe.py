#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how the Whisper STT service runs a transcription.

The service module is imported with ``pytest.importorskip`` so the suite is
skipped rather than failing collection when the optional Whisper dependencies
aren't installed.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The service raises ImportError (not ModuleNotFoundError) when its extra is absent.
pytest.importorskip("pipecat.services.whisper.stt", exc_type=ImportError)

from pipecat.frames.frames import (  # noqa: E402
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams  # noqa: E402
from pipecat.services.whisper.stt import WhisperSTTService  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402


class _LazyModel:
    """Stands in for ``WhisperModel``, whose ``transcribe`` returns a lazy generator.

    faster-whisper decodes each segment only when the generator is advanced, so
    the thread that consumes the generator is the thread that does the work.
    """

    supported_languages = ["en"]

    def __init__(self):
        self.decoding_threads: set[int] = set()
        self.received: list = []

    def transcribe(self, audio, **kwargs):
        self.received.append(audio)

        def segments():
            for text in ("Hello", "world."):
                self.decoding_threads.add(threading.get_ident())
                yield SimpleNamespace(text=text, no_speech_prob=0.0)

        return segments(), MagicMock()


def _build(model: _LazyModel, **kwargs) -> WhisperSTTService:
    with patch("pipecat.services.whisper.stt.WhisperModel", return_value=model):
        return WhisperSTTService(settings=WhisperSTTService.Settings(model="small.en"), **kwargs)


@pytest.mark.asyncio
async def test_segments_are_decoded_off_the_event_loop():
    """Decoding would otherwise stall every other task, audio transports included."""
    model = _LazyModel()
    service = _build(model)
    audio = b"\x00\x00" * 16000

    frames = [frame async for frame in service.run_stt(audio)]

    assert model.decoding_threads
    assert threading.get_ident() not in model.decoding_threads
    transcriptions = [frame for frame in frames if isinstance(frame, TranscriptionFrame)]
    assert [frame.text.split() for frame in transcriptions] == [["Hello", "world."]]


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000])
async def test_audio_reaches_whisper_at_16khz(sample_rate):
    """Whisper reads 16 kHz audio whatever rate the pipeline's input audio uses."""
    model = _LazyModel()
    service = _build(model, trailing_silence_secs=0)
    one_second = b"\x00\x00" * sample_rate

    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=one_second, sample_rate=sample_rate, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(audio_in_sample_rate=sample_rate),
    )

    assert [len(audio) for audio in model.received] == [16000]
