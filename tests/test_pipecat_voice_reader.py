#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that hears each user turn with Pipecat Voice Reader."""

import pytest

from pipecat.classifiers.tone.model import PipecatToneReaderModel
from pipecat.classifiers.voice.model import VOICE_REVISION, PipecatVoiceReaderModel
from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.processors.perception_reader import PerceptionReader, PerceptionSource
from pipecat.expressive.processors.tone_reader import PipecatToneSource
from pipecat.expressive.processors.voice_reader import PipecatVoiceReader, PipecatVoiceSource
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import run_test
from pipecat.utils import models

SAMPLE_RATE = 16000

TONES = ["positive", "neutral", "sad", "frustrated"]

READING = {
    "tone": {"positive": 0.1, "neutral": 0.2, "sad": 0.05, "frustrated": 0.65},
    "sarcasm": 0.3,
    "laughter": 0.05,
    "sad": 0.02,
    "upset": 0.7,
    "arousal": 0.4321,
    "speech_rate": 3.14,
}

HEARD = [
    PerceptionSignal("tone", labels=[("frustrated", 0.65), ("neutral", 0.2)]),
    PerceptionSignal("arousal", value=0.43),
    PerceptionSignal("laughter", value=0.05),
    PerceptionSignal("speech_rate", value=3.14),
    PerceptionSignal("sarcasm", value=0.3),
]


class FakeVoiceModel:
    """Answers every clip with READING, and remembers the clips it was given."""

    tones = TONES
    outputs = list(READING)
    sarcasm_threshold = 0.066

    def __init__(self, error: Exception | None = None):
        self.clips: list[bytes] = []
        self._error = error

    def read(self, audio: bytes):
        self.clips.append(audio)
        if self._error:
            raise self._error
        return READING


class FakeToneModel:
    """Stands in for Pipecat Tone's reader: every turn reads angry and sarcastic."""

    emotions = ["neutral", "happy", "angry", "sad"]
    moves = ["asks", "answers", "other"]

    def tag(self, text: str, previous: str | None = None):
        return {
            "emotion": {
                "label": "angry",
                "confidence": 0.71,
                "probabilities": {"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
            },
            "sarcasm": {"probability": 0.99},
        }


@pytest.fixture(autouse=True)
def _unshared_models(monkeypatch):
    """Each test loads its own model."""
    monkeypatch.setattr(models, "_models", {})


def _reader(monkeypatch, model: FakeVoiceModel, **kwargs) -> PipecatVoiceReader:
    """A reader whose model downloads as ``model``."""
    monkeypatch.setattr(PipecatVoiceReaderModel, "download", lambda **kwargs: model)
    return PipecatVoiceReader(**kwargs)


def _speech(text: str, secs: float = 2.0) -> list:
    """``secs`` seconds of the user's speech, then its transcription."""
    audio = bytes([1]) * (int(SAMPLE_RATE * secs) * 2)
    return [
        VADUserStartedSpeakingFrame(),
        InputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1),
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame(text=text, user_id="", timestamp="now"),
    ]


def _perception(frames) -> list:
    return [f.perception for f in frames if isinstance(f, TranscriptionFrame)]


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start(monkeypatch):
    down, _ = await run_test(_reader(monkeypatch, FakeVoiceModel()), frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.labels) for t in metadata[0].perception.signals] == [
        ("tone", TONES),
        ("arousal", []),
        ("laughter", []),
        ("speech_rate", []),
        ("sarcasm", []),
    ]
    assert all(term.description for term in metadata[0].perception.signals)


@pytest.mark.asyncio
async def test_what_the_model_heard_is_set_on_the_transcription_of_the_speech(monkeypatch):
    model = FakeVoiceModel()

    down, _ = await run_test(_reader(monkeypatch, model), frames_to_send=_speech("Oh, great."))

    assert len(model.clips) == 1
    assert _perception(down) == [HEARD]


@pytest.mark.asyncio
async def test_sadness_is_reported_when_asked_for(monkeypatch):
    down, _ = await run_test(
        _reader(monkeypatch, FakeVoiceModel(), sad=True), frames_to_send=_speech("Fine.")
    )

    assert _perception(down) == [[*HEARD, PerceptionSignal("sad", value=0.02)]]


@pytest.mark.asyncio
async def test_speech_too_short_for_the_model_isnt_read(monkeypatch):
    model = FakeVoiceModel()

    down, _ = await run_test(_reader(monkeypatch, model), frames_to_send=_speech("No.", secs=0.5))

    assert model.clips == []
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_the_voice_is_heard_beside_a_reading_of_the_words(monkeypatch):
    monkeypatch.setattr(PipecatToneReaderModel, "download", lambda **kwargs: FakeToneModel())
    monkeypatch.setattr(PipecatVoiceReaderModel, "download", lambda **kwargs: FakeVoiceModel())
    # Both report a ``sarcasm``, so the voice's is asked under a name of its own.
    voice = PipecatVoiceSource()
    questions = dict(voice.questions)
    questions["sarcasm_voice"] = questions.pop("sarcasm")
    reader = PerceptionReader(
        sources=[
            PipecatToneSource(),
            PerceptionSource(classifier=voice.classifier, questions=questions),
        ]
    )

    down, _ = await run_test(reader, frames_to_send=_speech("Oh, great."))

    assert _perception(down) == [
        [
            PerceptionSignal("emotion", labels=[("angry", 0.71), ("neutral", 0.2)]),
            PerceptionSignal("sarcasm", value=0.99),
            *HEARD[:-1],
            PerceptionSignal("sarcasm_voice", value=0.3),
        ]
    ]


def test_the_voice_and_the_words_cant_both_report_sarcasm_in_one_reader():
    with pytest.raises(ValueError, match="sarcasm"):
        PerceptionReader(sources=[PipecatToneSource(), PipecatVoiceSource()])


@pytest.mark.asyncio
async def test_a_failed_read_is_reported_and_the_transcription_still_passes(monkeypatch):
    down, up = await run_test(
        _reader(monkeypatch, FakeVoiceModel(error=RuntimeError("no session"))),
        frames_to_send=_speech("Hello."),
    )

    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_a_model_that_cant_load_leaves_the_reader_unusable_and_the_pipeline_running(
    monkeypatch,
):
    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatVoiceReaderModel, "download", fail)
    reader = PipecatVoiceReader()

    down, up = await run_test(reader, frames_to_send=_speech("Hello there."))

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_the_model_downloads_at_the_release_and_variant_asked_for(monkeypatch):
    asked = []

    def download(**kwargs):
        asked.append(kwargs)
        return FakeVoiceModel()

    monkeypatch.setattr(PipecatVoiceReaderModel, "download", download)
    for reader in (PipecatVoiceReader(), PipecatVoiceReader(variant="tiny", revision="rc9")):
        await run_test(reader, frames_to_send=[])

    assert [(a["revision"], a["variant"]) for a in asked] == [
        (VOICE_REVISION, "base"),
        ("rc9", "tiny"),
    ]
