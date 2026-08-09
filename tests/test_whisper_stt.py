#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.whisper import stt as whisper_stt
from pipecat.services.whisper.stt import Model, WhisperSTTService, WhisperSTTServiceMLX
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 16000
# 16-bit mono PCM: 2 bytes per sample.
ONE_SECOND_PCM = b"\x01\x02" * SAMPLE_RATE


def _make_service(monkeypatch, **kwargs) -> tuple[WhisperSTTService, list]:
    """Build a WhisperSTTService with stubbed model instances."""

    models = []

    class _Segment:
        no_speech_prob = 0.0

        def __init__(self, text: str):
            self.text = text

    class _FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.transcribe_calls = 0
            models.append(self)

        def transcribe(self, audio, *, language):
            self.transcribe_calls += 1
            text = (
                "final transcript"
                if self.model_name == Model.DISTIL_MEDIUM_EN.value
                else "interim transcript"
            )
            return iter([_Segment(text)]), None

    monkeypatch.setattr(whisper_stt, "WhisperModel", _FakeWhisperModel)
    return WhisperSTTService(sample_rate=SAMPLE_RATE, **kwargs), models


@pytest.mark.asyncio
async def test_interims_use_interim_model_and_final_uses_final_model(monkeypatch):
    service, models = _make_service(monkeypatch, interim_interval=0.2, interim_model=Model.BASE)

    interim = await service.run_interim_stt(ONE_SECOND_PCM)
    finals = [frame async for frame in service.run_stt(ONE_SECOND_PCM)]

    assert len(models) == 2
    assert models[1].model_name == Model.BASE.value
    assert interim == "interim transcript "
    assert len(finals) == 1
    assert isinstance(finals[0], TranscriptionFrame)
    assert finals[0].text == "final transcript "


@pytest.mark.asyncio
async def test_no_interim_model_loaded_when_disabled(monkeypatch):
    service, models = _make_service(monkeypatch)

    assert service._interim_model is None
    assert len(models) == 1


@pytest.mark.asyncio
async def test_short_audio_skips_interim_transcription(monkeypatch):
    service, models = _make_service(monkeypatch, interim_interval=0.2)
    short_audio = b"\x01\x02" * (SAMPLE_RATE // 2 - 1)

    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=short_audio, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.25),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert models[1].transcribe_calls == 0


def test_mlx_rejects_interim_transcriptions():
    with pytest.raises(ValueError, match="does not support interim transcriptions"):
        WhisperSTTServiceMLX(interim_interval=0.2)
