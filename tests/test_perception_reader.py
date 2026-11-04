#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that reads each user turn by asking a classifier."""

import asyncio

import pytest

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.processors.perception_reader import PerceptionQuestion, PerceptionReader
from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextAssistantTurnFrame,
    MetricsFrame,
    PerceptionFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test

QUESTIONS = {
    "tone": PerceptionQuestion(
        question=ChoiceQuestion(
            instructions="How does the user sound?",
            options={"calm": None, "frustrated": None, "confused": None},
        ),
        description="How the user sounded.",
    ),
    "hesitation": PerceptionQuestion(
        question=YesNoQuestion(instructions="Does the user sound unsure?"),
        description="The chance the user sounded unsure, from 0 to 1.",
    ),
    "energy": PerceptionQuestion(
        question=ScoreQuestion(instructions="How energetic is the user?", levels=["low", "high"]),
        description="From 0, low, to 1, high.",
    ),
}

RESULTS = {
    "tone": ChoiceResult(
        choice="frustrated",
        probabilities={"calm": 0.1, "frustrated": 0.7, "confused": 0.2},
        confidence=0.6,
    ),
    "hesitation": YesNoResult(probability=0.6432),
    "energy": ScoreResult(
        score=0.678,
        levels=[
            ScoreLevel(level="low", probability=0.32),
            ScoreLevel(level="high", probability=0.68),
        ],
        confidence=0.4,
    ),
}


class FakeClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        error: Exception | None = None,
        setup_error: Exception | None = None,
        delay: float = 0.0,
    ):
        super().__init__()
        self.asked: list[tuple] = []
        self._error = error
        self._setup_error = setup_error
        self._delay = delay

    async def setup(self, task_manager):
        await super().setup(task_manager)
        if self._setup_error:
            raise self._setup_error

    async def _ask(self, state, questions, images=()):
        self.asked.append((state, dict(questions)))
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return {name: RESULTS[name] for name in questions}, None


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start():
    reader = PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS)

    down, _ = await run_test(reader, frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.description, t.labels) for t in metadata[0].perception.signals] == [
        ("tone", "How the user sounded.", ["calm", "frustrated", "confused"]),
        ("hesitation", "The chance the user sounded unsure, from 0 to 1.", []),
        ("energy", "From 0, low, to 1, high.", []),
    ]


@pytest.mark.asyncio
async def test_each_answer_is_set_as_a_signal_on_its_transcription():
    down, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=[_transcription("I already gave you my order number.")],
        expected_down_frames=[PerceptionMetadataFrame, PerceptionFrame, TranscriptionFrame],
    )

    assert down[2].perception == [
        PerceptionSignal("tone", labels=[("frustrated", 0.7), ("confused", 0.2)]),
        PerceptionSignal("hesitation", value=0.64),
        PerceptionSignal("energy", value=0.68),
    ]


@pytest.mark.asyncio
async def test_a_choice_reports_at_most_its_max_labels():
    question = QUESTIONS["tone"].model_copy(update={"max_labels": 1})

    down, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions={"tone": question}),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, PerceptionFrame, TranscriptionFrame],
    )

    assert down[2].perception == [PerceptionSignal("tone", labels=[("frustrated", 0.7)])]


@pytest.mark.asyncio
async def test_a_turn_is_asked_about_with_the_bots_words_before_it():
    classifier = FakeClassifier()

    await run_test(
        PerceptionReader(classifier=classifier, questions=QUESTIONS),
        frames_to_send=[
            _transcription("Hi."),
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            _transcription("I already gave it to you."),
        ],
    )

    assert [state for state, _ in classifier.asked] == [
        {"user": "Hi."},
        {"assistant": "What's your order number?", "user": "I already gave it to you."},
    ]


@pytest.mark.asyncio
async def test_a_late_reading_lets_the_transcription_pass_unread():
    down, up = await run_test(
        PerceptionReader(classifier=FakeClassifier(delay=1.0), questions=QUESTIONS, timeout=0.05),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert not [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_failed_reading_is_reported_and_the_transcription_still_passes():
    down, up = await run_test(
        PerceptionReader(
            classifier=FakeClassifier(error=ClassifierError("down")), questions=QUESTIONS
        ),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_classifier_that_cant_be_set_up_leaves_the_reader_unusable():
    classifier = FakeClassifier(setup_error=ClassifierError("no model"))
    reader = PerceptionReader(classifier=classifier, questions=QUESTIONS)

    down, up = await run_test(reader, frames_to_send=[_transcription("Hello.")])

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [f.perception for f in down if isinstance(f, TranscriptionFrame)] == [None]
    assert classifier.asked == []


def test_a_name_that_cant_be_written_in_a_reading_is_refused():
    with pytest.raises(ValueError):
        PerceptionReader(
            classifier=FakeClassifier(), questions={"speech rate": QUESTIONS["energy"]}
        )


@pytest.mark.asyncio
async def test_the_classifiers_metrics_are_pushed_when_metrics_are_on():
    frames = [_transcription("Hello."), SleepFrame(sleep=0.2)]

    with_metrics, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    without_metrics, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
    )

    assert [f for f in with_metrics if isinstance(f, MetricsFrame)]
    assert not [f for f in without_metrics if isinstance(f, MetricsFrame)]


def _readings(frames) -> list:
    return [f.perception for f in frames if isinstance(f, PerceptionFrame)]


@pytest.mark.asyncio
async def test_a_text_reading_is_broadcast_both_ways():
    down, up = await run_test(
        PerceptionReader(
            classifier=FakeClassifier(), questions={"hesitation": QUESTIONS["hesitation"]}
        ),
        frames_to_send=[_transcription("Hello.")],
    )

    reading = [PerceptionSignal("hesitation", value=0.64)]
    assert _readings(down) == [reading]
    assert _readings(up) == [reading]
