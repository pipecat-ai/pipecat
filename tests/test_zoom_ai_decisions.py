#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for converting classifier questions and results to Zoom Decisions' format."""

import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.services.zoom_ai.decisions import (
    question_to_wire,
    questions_to_wire,
    result_from_wire,
)

_CHOICE = ChoiceQuestion(
    instructions="Which resolution?", options={"refund": None, "replacement": "A new one."}
)
_SCORE = ScoreQuestion(instructions="How urgent?", levels=["Low", "Medium", "High"])


def test_a_yes_no_question_is_a_predicate_with_its_criteria_in_the_instructions():
    question = YesNoQuestion(
        instructions="Did they agree?", yes="They said yes.", no="Anything else."
    )
    assert question_to_wire("agreed", question) == {
        "type": "predicate",
        "name": "agreed",
        "instructions": "Did they agree?\nYes means: They said yes.\nNo means: Anything else.",
    }


def test_choice_and_score_questions_keep_their_options_and_levels_in_order():
    assert questions_to_wire({"resolution": _CHOICE, "urgency": _SCORE}) == [
        {
            "type": "choice",
            "name": "resolution",
            "instructions": "Which resolution?",
            "choices": [
                {"value": "refund"},
                {"value": "replacement", "description": "A new one."},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent?",
            "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}],
        },
    ]


def test_structured_instructions_are_sent_as_json():
    question = YesNoQuestion(instructions={"question": "Is it late?", "order": "1001"})
    wire = question_to_wire("late", question)
    assert '"question": "Is it late?"' in wire["instructions"]


def test_a_predicate_answer_is_a_yes_no_result():
    answer = {"type": "predicate", "name": "agreed", "probability": 0.8}
    assert result_from_wire("agreed", YesNoQuestion(instructions="?"), answer) == YesNoResult(
        probability=0.8
    )


def test_a_choice_answer_with_probabilities_is_a_choice_result():
    answer = {
        "type": "choice",
        "choice": "refund",
        "probabilities": [
            {"value": "refund", "probability": 0.7},
            {"value": "replacement", "probability": 0.3},
        ],
    }
    assert result_from_wire("resolution", _CHOICE, answer) == ChoiceResult(
        choice="refund", probabilities={"refund": 0.7, "replacement": 0.3}, confidence=0.7
    )


def test_a_score_answer_maps_level_positions_back_to_the_levels():
    answer = {
        "type": "score",
        "score": 1.2,
        "probabilities": [
            {"value": 0, "probability": 0.1},
            {"value": 1, "probability": 0.6},
            {"value": 2, "probability": 0.3},
        ],
    }
    assert result_from_wire("urgency", _SCORE, answer) == ScoreResult(
        score=1.2,
        levels=[
            ScoreLevel(level="Low", probability=0.1),
            ScoreLevel(level="Medium", probability=0.6),
            ScoreLevel(level="High", probability=0.3),
        ],
        confidence=0.6,
    )


@pytest.mark.parametrize(
    "question,answer",
    [
        (_CHOICE, {"type": "refusal", "name": "resolution"}),
        (_CHOICE, {"type": "choice", "choice": "store_credit", "probability": 0.9}),
        (YesNoQuestion(instructions="?"), {"type": "predicate"}),
        (_SCORE, {"type": "score", "probabilities": []}),
    ],
)
def test_an_answer_that_doesnt_fit_its_question_raises(question, answer):
    with pytest.raises(ClassifierError):
        result_from_wire("q", question, answer)
