#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the classifier backed by Pipecat Tone's reader."""

import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.tone.classifier import PipecatToneClassifier
from pipecat.classifiers.tone.model import PipecatToneReaderModel
from pipecat.utils import models
from pipecat.utils.asyncio.task_manager import TaskManager

TAGS = {
    "emotion": {
        "label": "angry",
        "intensity": "low",
        "confidence": 0.71,
        "probabilities": {"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
    },
    "move": {
        "label": "answers",
        "confidence": 0.8,
        "probabilities": {"asks": 0.1, "answers": 0.8, "other": 0.1},
    },
    "sarcasm": {"sarcastic": True, "probability": 0.99},
    "backchannel": {"is_backchannel": False, "probability": 0.01},
}


class FakeReaderModel:
    emotions = ["neutral", "happy", "angry", "sad"]
    moves = ["asks", "answers", "other"]

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
    monkeypatch.setattr(models, "_models", {})


async def _classifier(monkeypatch, model: FakeReaderModel) -> PipecatToneClassifier:
    """A classifier set up with ``model`` as its download."""
    monkeypatch.setattr(PipecatToneReaderModel, "download", lambda **kwargs: model)
    classifier = PipecatToneClassifier()
    await classifier.setup(TaskManager())
    return classifier


@pytest.mark.asyncio
async def test_the_questions_come_from_the_loaded_model(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeReaderModel())

    questions = classifier.questions

    assert list(questions) == ["emotion", "move", "sarcasm", "backchannel"]
    assert list(questions["emotion"].options) == ["neutral", "happy", "angry", "sad"]
    assert list(questions["move"].options) == ["asks", "answers", "other"]
    assert isinstance(questions["sarcasm"], YesNoQuestion)


@pytest.mark.asyncio
async def test_nothing_is_answered_before_the_model_loads():
    classifier = PipecatToneClassifier()

    with pytest.raises(ClassifierError):
        _ = classifier.questions
    with pytest.raises(ClassifierError):
        await classifier.ask("Hello.", {"sarcasm": YesNoQuestion(instructions="Is it sarcastic?")})


@pytest.mark.asyncio
async def test_each_question_is_answered_from_one_read_of_the_turn(monkeypatch):
    model = FakeReaderModel()
    classifier = await _classifier(monkeypatch, model)

    results = await classifier.ask("Oh great.", classifier.questions)

    assert model.reads == [("Oh great.", None)]
    assert results["emotion"] == ChoiceResult(
        choice="angry", probabilities=TAGS["emotion"]["probabilities"], confidence=0.71
    )
    assert results["move"].choice == "answers"
    assert results["sarcasm"] == YesNoResult(probability=0.99)
    assert results["backchannel"] == YesNoResult(probability=0.01)


@pytest.mark.asyncio
async def test_a_turn_is_read_with_the_bots_words_before_it(monkeypatch):
    model = FakeReaderModel()
    classifier = await _classifier(monkeypatch, model)

    await classifier.ask(
        {"assistant": "Your next visit is in forty minutes.", "user": "Oh great."},
        {"sarcasm": classifier.questions["sarcasm"]},
    )

    assert model.reads == [("Oh great.", "Your next visit is in forty minutes.")]


@pytest.mark.asyncio
async def test_a_question_of_its_own_making_is_refused_before_reading(monkeypatch):
    model = FakeReaderModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask(
            "Oh great.",
            {"tone": ChoiceQuestion(instructions="How is the user?", options={"fine": None})},
        )

    assert model.reads == []


@pytest.mark.asyncio
async def test_a_state_that_isnt_a_turn_is_refused(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeReaderModel())

    with pytest.raises(ClassifierError):
        await classifier.ask({"transcript": "Oh great."}, classifier.questions)


@pytest.mark.asyncio
async def test_a_failed_read_raises_a_classifier_error(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeReaderModel(error=RuntimeError("no session")))

    with pytest.raises(ClassifierError):
        await classifier.ask("Hello.", classifier.questions)


@pytest.mark.asyncio
async def test_a_model_that_cant_load_fails_setup(monkeypatch):
    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatToneReaderModel, "download", fail)

    with pytest.raises(ClassifierError):
        await PipecatToneClassifier().setup(TaskManager())


@pytest.mark.asyncio
async def test_every_classifier_shares_the_loaded_model(monkeypatch):
    loads = []

    def download(**kwargs):
        loads.append(kwargs)
        return FakeReaderModel()

    monkeypatch.setattr(PipecatToneReaderModel, "download", download)
    for classifier in (PipecatToneClassifier(), PipecatToneClassifier()):
        await classifier.setup(TaskManager())

    assert len(loads) == 1
