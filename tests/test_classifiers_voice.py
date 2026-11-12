#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the classifier backed by Pipecat Voice Reader, and for its model."""

import json

import numpy as np
import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceResult,
    ClassifierAudio,
    ClassifierError,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.voice import model as voice_model
from pipecat.classifiers.voice.classifier import PipecatVoiceClassifier
from pipecat.classifiers.voice.model import PipecatVoiceReaderModel
from pipecat.utils import models
from pipecat.utils.asyncio.task_manager import TaskManager

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


class FakeVoiceModel:
    """Answers every clip with READING, and remembers the clips it was given."""

    sarcasm_threshold = 0.066

    def __init__(self, *, tones: list[str] = TONES, error: Exception | None = None):
        self.tones = tones
        self.outputs = list(READING)
        self.clips: list[bytes] = []
        self._error = error

    def read(self, audio: bytes):
        self.clips.append(audio)
        if self._error:
            raise self._error
        return READING


@pytest.fixture(autouse=True)
def _unshared_models(monkeypatch):
    """Each test loads its own model."""
    monkeypatch.setattr(models, "_models", {})


def _pcm(secs: float, *, rate: int = SAMPLE_RATE, hz: float = 220.0, offset: float = 0.0) -> bytes:
    """``secs`` seconds of a tone as 16-bit mono PCM, lifted off zero by ``offset``."""
    t = np.arange(int(secs * rate)) / rate
    wave = 0.3 * np.sin(2 * np.pi * hz * t) + offset
    return (wave * 32767).astype(np.int16).tobytes()


def _clip(secs: float, *, rate: int = SAMPLE_RATE) -> ClassifierAudio:
    return ClassifierAudio(audio=_pcm(secs, rate=rate), sample_rate=rate)


async def _classifier(monkeypatch, model: FakeVoiceModel, **kwargs) -> PipecatVoiceClassifier:
    """A classifier set up with ``model`` as its download."""
    monkeypatch.setattr(PipecatVoiceReaderModel, "download", lambda **kwargs: model)
    classifier = PipecatVoiceClassifier(**kwargs)
    await classifier.setup(TaskManager())
    return classifier


def test_the_questions_are_known_before_the_model_loads():
    questions = PipecatVoiceClassifier().questions

    assert list(questions) == [
        "tone",
        "sarcasm",
        "laughter",
        "sad",
        "arousal",
        "upset",
        "speech_rate",
    ]
    assert list(questions["tone"].options) == TONES
    assert isinstance(questions["sarcasm"], YesNoQuestion)
    assert isinstance(questions["speech_rate"], ScoreQuestion)


@pytest.mark.asyncio
async def test_nothing_is_answered_before_the_model_loads():
    classifier = PipecatVoiceClassifier()

    with pytest.raises(ClassifierError):
        _ = classifier.sarcasm_threshold
    with pytest.raises(ClassifierError):
        await classifier.ask({}, classifier.questions, inputs=[_clip(2)])


@pytest.mark.asyncio
async def test_each_question_is_answered_from_one_read_of_the_clip(monkeypatch):
    model = FakeVoiceModel()
    classifier = await _classifier(monkeypatch, model)

    results = await classifier.ask({}, classifier.questions, inputs=[_clip(2)])

    assert model.clips == [_pcm(2)]
    assert results["tone"] == ChoiceResult(
        choice="frustrated", probabilities=READING["tone"], confidence=0.65
    )
    assert results["sarcasm"] == YesNoResult(probability=0.3)
    assert results["laughter"] == YesNoResult(probability=0.05)
    assert results["sad"] == YesNoResult(probability=0.02)
    arousal, upset = results["arousal"], results["upset"]
    assert isinstance(arousal, ScoreResult) and isinstance(upset, ScoreResult)
    assert arousal.score == pytest.approx(0.4321)
    assert upset.score == pytest.approx(0.7)
    assert upset.probability("unmistakably") == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_a_speech_rate_sits_between_the_levels_around_it(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeVoiceModel())

    results = await classifier.score(
        {}, {"rate": classifier.questions["speech_rate"]}, inputs=[_clip(2)]
    )

    rate = results["rate"]
    assert rate.score == pytest.approx(3.14)
    assert rate.probability("3 syllables per second") == pytest.approx(0.86)
    assert rate.probability("4 syllables per second") == pytest.approx(0.14)
    assert rate.confidence == pytest.approx(0.86)


@pytest.mark.asyncio
async def test_a_question_goes_by_what_it_asks_not_its_name(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeVoiceModel())

    results = await classifier.ask(
        {}, {"sarcasm_voice": classifier.questions["sarcasm"]}, inputs=[_clip(2)]
    )

    assert results == {"sarcasm_voice": YesNoResult(probability=0.3)}


@pytest.mark.asyncio
async def test_a_clip_at_another_rate_is_resampled_for_the_model(monkeypatch):
    model = FakeVoiceModel()
    classifier = await _classifier(monkeypatch, model)

    await classifier.ask({}, classifier.questions, inputs=[_clip(2, rate=48000)])

    assert len(model.clips[0]) == pytest.approx(2 * SAMPLE_RATE * 2, abs=SAMPLE_RATE // 50)


@pytest.mark.asyncio
@pytest.mark.parametrize("secs", [0.5, 9.0])
async def test_a_clip_shorter_or_longer_than_the_model_hears_is_refused(monkeypatch, secs):
    model = FakeVoiceModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask({}, classifier.questions, inputs=[_clip(secs)])

    assert model.clips == []


@pytest.mark.asyncio
@pytest.mark.parametrize("clips", [0, 2])
async def test_it_answers_about_exactly_one_clip(monkeypatch, clips):
    model = FakeVoiceModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask({}, classifier.questions, inputs=[_clip(2)] * clips)

    assert model.clips == []


@pytest.mark.asyncio
async def test_a_question_of_its_own_making_is_refused_before_reading(monkeypatch):
    model = FakeVoiceModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask(
            {},
            {"hurry": YesNoQuestion(instructions="Is the user in a hurry?")},
            inputs=[_clip(2)],
        )

    assert model.clips == []


@pytest.mark.asyncio
async def test_a_question_the_model_has_no_output_for_is_refused_before_reading(monkeypatch):
    model = FakeVoiceModel()
    model.outputs.remove("sad")
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError, match="sad"):
        await classifier.ask({}, {"sad": classifier.questions["sad"]}, inputs=[_clip(2)])

    assert model.clips == []


@pytest.mark.asyncio
async def test_a_failed_read_raises_a_classifier_error(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeVoiceModel(error=RuntimeError("no session")))

    with pytest.raises(ClassifierError):
        await classifier.ask({}, classifier.questions, inputs=[_clip(2)])


@pytest.mark.asyncio
async def test_a_model_with_other_tones_isnt_set_up(monkeypatch):
    with pytest.raises(ClassifierError):
        await _classifier(monkeypatch, FakeVoiceModel(tones=["calm", "tense"]))


@pytest.mark.asyncio
async def test_a_model_that_cant_load_fails_setup(monkeypatch):
    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatVoiceReaderModel, "download", fail)

    with pytest.raises(ClassifierError):
        await PipecatVoiceClassifier().setup(TaskManager())


@pytest.mark.asyncio
async def test_the_sarcasm_threshold_is_the_loaded_models(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeVoiceModel())

    assert classifier.sarcasm_threshold == 0.066


@pytest.mark.asyncio
async def test_classifiers_share_the_loaded_model_of_their_variant(monkeypatch):
    loads = []

    def download(**kwargs):
        loads.append(kwargs["variant"])
        return FakeVoiceModel()

    monkeypatch.setattr(PipecatVoiceReaderModel, "download", download)
    for variant in ("base", "base", "tiny"):
        await PipecatVoiceClassifier(variant=variant).setup(TaskManager())

    assert loads == ["base", "tiny"]


def test_the_model_is_named_by_its_release_and_variant_or_its_folder(tmp_path):
    assert PipecatVoiceClassifier(variant="tiny", revision="rc9").model == (
        "pipecat-ai/voice-reader@rc9/tiny"
    )
    assert PipecatVoiceClassifier(model_dir=tmp_path).model == str(tmp_path)


CONFIG = {
    "window_secs": 8.0,
    "read_secs": [4.0, 6.0, 8.0],
    "sample_rate": SAMPLE_RATE,
    "tones": TONES,
    "outputs": {
        "tone": {"kind": "distribution"},
        "sarcasm": {"kind": "chance", "threshold": 0.066},
        "laughter": {"kind": "chance"},
        "sad": {"kind": "chance"},
        "upset": {"kind": "level"},
        "arousal": {"kind": "level"},
        "speech_rate": {"kind": "rate"},
    },
}


class FakeSession:
    """Stands in for the ONNX session: gives READING, and remembers the features it was run on."""

    def __init__(self, path, options, providers):
        self.path = path
        self.features: list[np.ndarray] = []

    def run(self, output_names, inputs):
        self.features.append(inputs["input_features"])
        tone, *numbers = READING.values()
        return [
            np.array([list(tone.values())], dtype=np.float32),
            *(np.array([number], dtype=np.float32) for number in numbers),
        ]


def _model(monkeypatch, tmp_path, **config) -> tuple[PipecatVoiceReaderModel, FakeSession]:
    """A model loaded from a folder with CONFIG, changed by ``config``, and its session."""
    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **config}), encoding="utf-8")
    monkeypatch.setattr(voice_model.ort, "InferenceSession", FakeSession)
    model = PipecatVoiceReaderModel(tmp_path)
    return model, model._session


def test_the_model_names_each_value_by_its_output(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path)

    reading = model.read(_pcm(2))

    assert session.path == str(tmp_path / "model.int8.onnx")
    assert model.tones == TONES
    assert model.outputs == list(READING)
    assert reading["tone"] == pytest.approx(READING["tone"])
    assert {name: value for name, value in reading.items() if name != "tone"} == pytest.approx(
        {name: value for name, value in READING.items() if name != "tone"}
    )


@pytest.mark.parametrize("secs, frames", [(1.0, 400), (4.0, 400), (4.5, 600), (6.5, 800)])
def test_a_clip_is_read_in_the_shortest_length_that_holds_it(monkeypatch, tmp_path, secs, frames):
    model, session = _model(monkeypatch, tmp_path)

    model.read(_pcm(secs))

    assert session.features[0].shape == (1, 80, frames)


def test_a_model_that_names_no_lengths_reads_its_whole_window(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path, read_secs=None)

    model.read(_pcm(1))

    assert session.features[0].shape == (1, 80, 800)


def test_only_the_last_8_seconds_of_a_clip_are_read(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path)

    model.read(_pcm(3, hz=660.0) + _pcm(8))
    model.read(_pcm(8))

    assert np.array_equal(*session.features)


def test_a_clip_is_read_without_its_dc_offset(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path)

    model.read(_pcm(2, offset=0.05))

    # The silence after the clip's 200 frames stays at the features' floor.
    (features,) = session.features
    assert (features[0, :, 210:] == features.min()).all()


def test_the_sarcasm_threshold_comes_from_the_models_config(monkeypatch, tmp_path):
    model, _ = _model(monkeypatch, tmp_path)

    assert model.sarcasm_threshold == 0.066


@pytest.mark.parametrize(
    "config", [{"window_secs": 10.0}, {"outputs": ["tone", "sarcasm", "laughter"]}]
)
def test_a_model_the_class_doesnt_read_is_refused(monkeypatch, tmp_path, config):
    with pytest.raises(ValueError):
        _model(monkeypatch, tmp_path, **config)


def _release(monkeypatch, tmp_path) -> dict:
    """A release whose index names a folder for each variant, and what is asked of the Hub."""
    index = tmp_path / "config.json"
    index.write_text(
        json.dumps({"variants": {"base": {"folder": "big"}, "tiny": {"folder": "small"}}}),
        encoding="utf-8",
    )
    seen: dict = {}

    def hf_hub_download(repo, filename, revision):
        seen.update(index=(repo, filename, revision))
        return str(index)

    def snapshot_download(repo, revision, allow_patterns):
        seen.update(repo=repo, revision=revision, patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(voice_model, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(voice_model, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        PipecatVoiceReaderModel,
        "__init__",
        lambda self, model_dir, precision, threads: seen.update(dir=model_dir),
    )
    return seen


def test_the_model_downloads_its_files_from_the_folder_the_release_names(monkeypatch, tmp_path):
    seen = _release(monkeypatch, tmp_path)

    PipecatVoiceReaderModel.download(revision="rc9", variant="tiny")

    assert seen == {
        "index": ("pipecat-ai/voice-reader", "config.json", "rc9"),
        "repo": "pipecat-ai/voice-reader",
        "revision": "rc9",
        "patterns": ["small/config.json", "small/model.int8.onnx"],
        "dir": tmp_path / "small",
    }


def test_a_variant_the_release_doesnt_have_is_refused(monkeypatch, tmp_path):
    _release(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="huge"):
        PipecatVoiceReaderModel.download(variant="huge")


@pytest.mark.asyncio
async def test_a_folder_of_your_own_loads_in_place_of_a_download(monkeypatch, tmp_path):
    def download(**kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(PipecatVoiceReaderModel, "download", download)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    monkeypatch.setattr(voice_model.ort, "InferenceSession", FakeSession)
    classifier = PipecatVoiceClassifier(model_dir=tmp_path, precision="fp32")
    await classifier.setup(TaskManager())

    results = await classifier.ask({}, {"tone": classifier.questions["tone"]}, inputs=[_clip(2)])

    assert results["tone"].choice == "frustrated"
