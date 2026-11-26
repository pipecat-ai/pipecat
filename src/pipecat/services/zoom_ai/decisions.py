#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Zoom Decisions questions and answers in Pipecat's classifier types.

Pipecat asks Decisions-style questions as
:data:`~pipecat.classifiers.base_classifier.ClassifierQuestion` values, by
name, and reads the answers as
:data:`~pipecat.classifiers.base_classifier.ClassifierResult` values. These
functions convert between those types and the question and answer format of
the Zoom Decisions API, for everything in Pipecat that asks it.

See https://developers.zoom.us/docs/ai-services/decisions/
"""

import json
from collections.abc import Mapping
from typing import Any

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierQuestion,
    ClassifierResult,
    ScoreLevel,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)


def questions_to_wire(questions: Mapping[str, ClassifierQuestion]) -> list[dict[str, Any]]:
    """The questions in Zoom's format, in order, each named by its key.

    Args:
        questions: The questions, by name.

    Returns:
        One Zoom question per entry.
    """
    return [question_to_wire(name, question) for name, question in questions.items()]


def question_to_wire(name: str, question: ClassifierQuestion) -> dict[str, Any]:
    """One question in Zoom's format.

    A :class:`~pipecat.classifiers.base_classifier.YesNoQuestion` becomes a
    ``predicate``, a :class:`~pipecat.classifiers.base_classifier.ChoiceQuestion`
    a ``choice`` and a :class:`~pipecat.classifiers.base_classifier.ScoreQuestion`
    a ``score``. Structured instructions, descriptions and levels are sent as
    JSON.

    Args:
        name: Names the answer.
        question: The question.

    Returns:
        The question as Zoom takes it.
    """
    if isinstance(question, YesNoQuestion):
        # A predicate has no field for what counts as each answer, so it goes
        # in the instructions.
        lines = [_text(question.instructions)]
        if question.yes is not None:
            lines.append(f"Yes means: {_text(question.yes)}")
        if question.no is not None:
            lines.append(f"No means: {_text(question.no)}")
        return {"type": "predicate", "name": name, "instructions": "\n".join(lines)}
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "name": name,
            "instructions": _text(question.instructions),
            "choices": [
                {"value": option}
                if description is None
                else {"value": option, "description": _text(description)}
                for option, description in question.options.items()
            ],
        }
    return {
        "type": "score",
        "name": name,
        "instructions": _text(question.instructions),
        "levels": [{"label": _text(level)} for level in question.levels],
    }


def result_from_wire(
    name: str, question: ClassifierQuestion, answer: Mapping[str, Any]
) -> ClassifierResult:
    """The result for one question, from Zoom's answer to it.

    Args:
        name: The question's name.
        question: The question that was asked.
        answer: Zoom's answer to it.

    Returns:
        The result of the type the question calls for.

    Raises:
        ClassifierError: If Zoom refused the question, or the answer doesn't
            fit it.
    """
    # Zoom gives no confidence. A choice's is the chosen option's probability,
    # and a score's the likeliest level's.
    if answer.get("type") == "refusal":
        raise ClassifierError(f"Zoom Decisions refused to answer {name!r}")
    if isinstance(question, YesNoQuestion):
        return YesNoResult(probability=_number(answer, "probability"))
    if isinstance(question, ChoiceQuestion):
        choice = str(answer.get("choice", ""))
        if choice not in question.options:
            raise ClassifierError(f"Zoom Decisions chose {choice!r}, which is not an option")
        probabilities = _probabilities(answer, list(question.options))
        return ChoiceResult(
            choice=choice, probabilities=probabilities, confidence=probabilities[choice]
        )
    # Zoom keys level probabilities by position.
    positions = [str(index) for index in range(len(question.levels))]
    probabilities = _probabilities(answer, positions)
    return ScoreResult(
        score=_number(answer, "score"),
        levels=[
            ScoreLevel(level=level, probability=probabilities[position])
            for level, position in zip(question.levels, positions)
        ],
        confidence=max(probabilities.values()),
    )


def _text(value: Any) -> str:
    """Text for Zoom: a string as is, structured data as JSON."""
    return value if isinstance(value, str) else json.dumps(value, indent=2)


def _number(answer: Mapping[str, Any], key: str) -> float:
    """The number Zoom wrote under ``key``, or a ClassifierError if it did not."""
    try:
        return float(answer[key])
    except (KeyError, TypeError, ValueError) as e:
        raise ClassifierError(f"Zoom Decisions answer has no usable '{key}'") from e


def _probabilities(answer: Mapping[str, Any], keys: list[str]) -> dict[str, float]:
    """The probability Zoom gave for each key, 0 for the ones it left out.

    Zoom lists probabilities as ``{"value", "probability"}`` items, where the
    value is a choice's option or a level's position.
    """
    given = answer.get("probabilities")
    by_value: dict[str, Any] = {
        str(item.get("value")): item.get("probability")
        for item in (given if isinstance(given, list) else [])
        if isinstance(item, dict)
    }
    try:
        return {key: float(by_value.get(key, 0.0)) for key in keys}
    except (TypeError, ValueError) as e:
        raise ClassifierError(f"Zoom Decisions answer has an unusable probability: {e}") from e
