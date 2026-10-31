#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that reads each user turn with Pipecat Tone."""

import pytest

from pipecat.expressive.perception import PerceptionSignal
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
)
from pipecat.processors.tone import _models
from pipecat.processors.tone.reader import PipecatToneReader
from pipecat.processors.tone.reader import model as reader_model
from pipecat.processors.tone.reader.model import PipecatToneReaderModel
from pipecat.tests.utils import run_test

TAGS = {
    "emotion": {
        "label": "angry",
        "intensity": "low",
        "confidence": 0.71,
        "probabilities": {"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
    },
    "move": {"label": "other", "confidence": 0.92, "probabilities": {}},
    "sarcasm": {"sarcastic": True, "probability": 0.99},
    "backchannel": {"is_backchannel": False, "probability": 0.01},
}


class FakeReaderModel:
    emotions = ["neutral", "happy", "angry", "sad"]

    def __init__(self, error: Exception | None = None):
        self.reads: list[tuple[str, str | None]] = []
        self._error = error

    def tag(self, text: str, previous: str | None = None):
        self.reads.append((text, previous))
        if self._error:
            raise self._error
        return TAGS


@pytest.fixture(autouse=True)
def _unshared_models(monkeypatch):
    """Each test loads its own model."""
    monkeypatch.setattr(_models, "_models", {})


def _reader(monkeypatch, model: FakeReaderModel) -> PipecatToneReader:
    """A reader whose model downloads as ``model``."""
    monkeypatch.setattr(PipecatToneReaderModel, "download", lambda **kwargs: model)
    return PipecatToneReader()


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start(monkeypatch):
    down, _ = await run_test(_reader(monkeypatch, FakeReaderModel()), frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.labels) for t in metadata[0].perception.signals] == [
        ("emotion", ["neutral", "happy", "angry", "sad"]),
        ("sarcasm", []),
    ]


@pytest.mark.asyncio
async def test_the_reading_is_set_on_its_transcription(monkeypatch):
    down, _ = await run_test(
        _reader(monkeypatch, FakeReaderModel()),
        frames_to_send=[_transcription("Oh great, a forty minute wait.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception == [
        PerceptionSignal("emotion", labels=[("angry", 0.71), ("neutral", 0.2)]),
        PerceptionSignal("sarcasm", value=0.99),
    ]


@pytest.mark.asyncio
async def test_a_turn_is_read_against_the_bots_words_before_it(monkeypatch):
    model = FakeReaderModel()

    await run_test(
        _reader(monkeypatch, model),
        frames_to_send=[
            LLMContextAssistantTurnFrame(
                text="The next visit is in forty minutes. Sorry.", timestamp="now"
            ),
            _transcription("Oh great."),
        ],
    )

    assert model.reads == [("Oh great.", "The next visit is in forty minutes. Sorry.")]


@pytest.mark.asyncio
async def test_interim_transcriptions_are_not_read(monkeypatch):
    model = FakeReaderModel()

    await run_test(
        _reader(monkeypatch, model),
        frames_to_send=[InterimTranscriptionFrame(text="Oh", user_id="", timestamp="now")],
    )

    assert model.reads == []


@pytest.mark.asyncio
async def test_a_failed_read_is_reported_and_the_transcription_still_passes(monkeypatch):
    down, up = await run_test(
        _reader(monkeypatch, FakeReaderModel(error=RuntimeError("no session"))),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert down[1].perception is None


@pytest.mark.asyncio
async def test_a_model_that_cant_load_leaves_the_reader_unusable_and_the_pipeline_running(
    monkeypatch,
):
    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatToneReaderModel, "download", fail)
    reader = PipecatToneReader()

    down, up = await run_test(reader, frames_to_send=[_transcription("Hello there.")])

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [f.perception for f in down if isinstance(f, TranscriptionFrame)] == [None]


@pytest.mark.asyncio
async def test_the_model_downloads_at_the_release_asked_for(monkeypatch):
    asked = []

    def download(**kwargs):
        asked.append(kwargs)
        return FakeReaderModel()

    monkeypatch.setattr(PipecatToneReaderModel, "download", download)
    for reader in (PipecatToneReader(), PipecatToneReader(revision="reader-rc9")):
        await run_test(reader, frames_to_send=[])

    assert [a["revision"] for a in asked] == ["reader-rc2", "reader-rc9"]


def test_the_model_downloads_its_files_from_the_reader_folder(monkeypatch, tmp_path):
    seen = {}

    def snapshot_download(repo, revision, allow_patterns):
        seen.update(repo=repo, revision=revision, patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(reader_model, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        PipecatToneReaderModel,
        "__init__",
        lambda self, model_dir, precision, threads: seen.update(dir=model_dir),
    )

    PipecatToneReaderModel.download()

    assert seen == {
        "repo": "pipecat-ai/tone-models",
        "revision": "reader-rc2",
        "patterns": ["reader/config.json", "reader/tokenizer.json", "reader/model_fp16.onnx"],
        "dir": tmp_path / "reader",
    }


@pytest.mark.asyncio
async def test_a_folder_of_your_own_loads_in_place_of_a_download(monkeypatch, tmp_path):
    loaded = []

    def fail(**kwargs):
        raise OSError("no network")

    def load(self, model_dir, precision, threads):
        loaded.append((model_dir, precision))
        self.config = {"emotions": ["neutral", "happy"]}

    monkeypatch.setattr(PipecatToneReaderModel, "download", fail)
    monkeypatch.setattr(PipecatToneReaderModel, "__init__", load)

    down, _ = await run_test(
        PipecatToneReader(model_dir=str(tmp_path), precision="fp32"), frames_to_send=[]
    )

    assert loaded == [(tmp_path, "fp32")]
    assert [f for f in down if isinstance(f, PerceptionMetadataFrame)]
