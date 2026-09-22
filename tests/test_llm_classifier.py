#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMClassifier.

A scripted LLM service stands in for the model: ``run_inference()`` returns
the next scripted reply, so the tests exercise the question rendering, the
JSON parsing and the results.
"""

import asyncio

import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ClassifierError,
    ScoreQuestion,
    YesNoQuestion,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService


class _ScriptedLLM(LLMService):
    """Answers run_inference() with the next scripted reply and records the request."""

    def __init__(self, *replies: str | None):
        super().__init__()
        self._replies = list(replies)
        self.requests: list[tuple[str, str | None]] = []
        self.schemas: list[dict | None] = []

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
    ) -> str | None:
        content = context.messages[-1]["content"]
        assert isinstance(content, str)
        self.requests.append((content, system_instruction))
        self.schemas.append(response_schema)
        return self._replies.pop(0)


class _NoInferenceLLM(LLMService):
    pass


def _classifier(*replies: str | None, **kwargs) -> tuple[LLMClassifier, _ScriptedLLM]:
    llm = _ScriptedLLM(*replies)
    return LLMClassifier(llm=llm, **kwargs), llm


@pytest.mark.asyncio
async def test_yes_no_answers_from_the_json_reply():
    classifier, _ = _classifier('{"greeting": {"probability": 0.93}}')
    results = await classifier.yes_no(
        "Hello?", {"greeting": YesNoQuestion(instructions="a greeting?")}
    )
    assert results["greeting"].probability == 0.93


@pytest.mark.asyncio
async def test_choice_answers_with_a_probability_per_option():
    classifier, _ = _classifier(
        '{"turn": {"choice": "short", "probabilities": {"short": 0.8, "complete": 0.2}}}'
    )
    question = ChoiceQuestion(
        instructions="is the turn over?",
        options={"complete": "finished", "short": "cut off", "long": "needs time"},
    )
    results = await classifier.choice("I'd like to", {"turn": question})

    assert results["turn"].choice == "short"
    assert results["turn"].probabilities == {"complete": 0.2, "short": 0.8, "long": 0.0}
    assert results["turn"].confidence == 0.8


@pytest.mark.asyncio
async def test_score_answers_with_the_nearest_level():
    classifier, _ = _classifier('{"mood": {"score": 1.8, "confidence": 0.7}}')
    question = ScoreQuestion(instructions="how upset?", rubric=["calm", "impatient", "frustrated"])
    results = await classifier.score("This is the third time!", {"mood": question})

    assert results["mood"].score == 1.8
    assert results["mood"].confidence == 0.7
    assert results["mood"].probabilities == {"calm": 0.0, "impatient": 0.0, "frustrated": 0.7}


@pytest.mark.asyncio
async def test_several_questions_go_in_one_call():
    classifier, llm = _classifier(
        '{"greeting": {"probability": 0.9}, "mood": {"score": 0.0, "confidence": 0.8}}'
    )
    results = await classifier.ask(
        "Hello there!",
        {
            "greeting": YesNoQuestion(instructions="a greeting?"),
            "mood": ScoreQuestion(instructions="how upset?", rubric=["calm", "upset"]),
        },
    )

    assert len(llm.requests) == 1
    assert results["greeting"].probability == 0.9
    assert results["mood"].score == 0.0


@pytest.mark.asyncio
async def test_the_question_is_rendered_with_its_options_and_the_reply_shape():
    classifier, llm = _classifier('{"which": {"choice": "a"}}')
    await classifier.choice(
        {"assistant": "Where to?", "user": "japan"},
        {"which": ChoiceQuestion(instructions="which?", options={"a": "first", "b": None})},
    )

    message, instructions = llm.requests[0]
    assert message.startswith('Input:\n{\n  "assistant": "Where to?"')
    assert 'Question "which": which?' in message
    assert "- a: first\n- b\n" in message
    assert 'Reply with one JSON object of this shape: {"which": <answer to "which">}' in message
    assert instructions is not None and "single JSON object" in instructions


@pytest.mark.asyncio
async def test_yes_and_no_meanings_are_written_into_the_question():
    classifier, llm = _classifier('{"q": {"probability": 0.1}}')
    await classifier.yes_no(
        "Hello?",
        {"q": YesNoQuestion(instructions="is this a voicemail?", yes="a recording", no="a person")},
    )
    message, _ = llm.requests[0]
    assert "Yes means: a recording" in message
    assert "No means: a person" in message


@pytest.mark.asyncio
async def test_custom_instructions_reach_the_llm():
    classifier, llm = _classifier('{"q": {"probability": 0.5}}', instructions="Decide.")
    await classifier.yes_no("hi", {"q": YesNoQuestion(instructions="?")})
    assert llm.requests[0][1] == "Decide."


@pytest.mark.asyncio
async def test_a_fenced_reply_is_parsed():
    classifier, _ = _classifier('Sure:\n```json\n{"q": {"probability": 0.4}}\n```\n')
    results = await classifier.yes_no("hi", {"q": YesNoQuestion(instructions="?")})
    assert results["q"].probability == 0.4


@pytest.mark.asyncio
async def test_the_reply_schema_has_one_answer_per_question():
    classifier, llm = _classifier(
        '{"q": {"probability": 0.5}, "w": {"choice": "a", "probabilities": {"a": 1, "b": 0}}}'
    )
    await classifier.ask(
        "hi",
        {
            "q": YesNoQuestion(instructions="?"),
            "w": ChoiceQuestion(instructions="?", options={"a": "", "b": None}),
        },
    )

    schema = llm.schemas[0]
    assert schema is not None
    assert schema["required"] == ["q", "w"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["q"]["required"] == ["probability"]
    choice = schema["properties"]["w"]["properties"]
    assert choice["choice"]["enum"] == ["a", "b"]
    assert choice["probabilities"]["required"] == ["a", "b"]


@pytest.mark.asyncio
async def test_the_score_schema_asks_for_a_score_and_a_confidence():
    classifier, llm = _classifier('{"mood": {"score": 1, "confidence": 0.5}}')
    await classifier.score("hi", {"mood": ScoreQuestion(instructions="?", rubric=["a", "b"])})

    schema = llm.schemas[0]
    assert schema is not None
    assert schema["properties"]["mood"]["required"] == ["score", "confidence"]


@pytest.mark.asyncio
async def test_a_choice_without_probabilities_has_no_confidence():
    classifier, _ = _classifier('{"q": {"choice": "b"}}')
    results = await classifier.choice(
        "hi", {"q": ChoiceQuestion(instructions="?", options={"a": "", "b": ""})}
    )
    assert results["q"].choice == "b"
    assert results["q"].confidence == 0.0
    assert results["q"].probabilities == {"a": 0.0, "b": 0.0}


@pytest.mark.asyncio
async def test_a_lone_answer_without_its_name_is_accepted():
    classifier, _ = _classifier('{"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}')
    results = await classifier.choice(
        "hi", {"q": ChoiceQuestion(instructions="?", options={"a": "", "b": ""})}
    )
    assert results["q"].choice == "b"


@pytest.mark.asyncio
async def test_a_text_reply_is_an_error():
    classifier, _ = _classifier("I think yes.")
    with pytest.raises(ClassifierError, match="JSON object"):
        await classifier.yes_no("hi", {"q": YesNoQuestion(instructions="?")})


@pytest.mark.asyncio
async def test_a_missing_answer_is_an_error():
    classifier, _ = _classifier('{"other": {"probability": 0.5}}')
    with pytest.raises(ClassifierError, match="no answer for 'a'"):
        await classifier.yes_no(
            "hi", {"a": YesNoQuestion(instructions="?"), "b": YesNoQuestion(instructions="?")}
        )


@pytest.mark.asyncio
async def test_a_choice_outside_the_options_is_an_error():
    classifier, _ = _classifier('{"q": {"choice": "c"}}')
    with pytest.raises(ClassifierError, match="not an option"):
        await classifier.choice(
            "hi", {"q": ChoiceQuestion(instructions="?", options={"a": "", "b": ""})}
        )


@pytest.mark.asyncio
async def test_a_service_without_run_inference_is_an_error():
    classifier = LLMClassifier(llm=_NoInferenceLLM())
    with pytest.raises(ClassifierError, match="one-shot"):
        await classifier.yes_no("hi", {"q": YesNoQuestion(instructions="?")})


@pytest.mark.asyncio
async def test_every_call_reports_its_time_without_tokens():
    classifier, _ = _classifier('{"answer": {"probability": 0.9}}')
    reported = asyncio.Event()
    seen: list = []

    @classifier.event_handler("on_metrics")
    async def on_metrics(classifier, data):
        seen.extend(data)
        reported.set()

    await classifier.yes_no("hello", {"answer": YesNoQuestion(instructions="a greeting?")})
    await asyncio.wait_for(reported.wait(), 1)

    (processing,) = seen
    assert isinstance(processing, ProcessingMetricsData)
    assert processing.processor == classifier.name
    assert processing.model is None
    assert processing.value >= 0
