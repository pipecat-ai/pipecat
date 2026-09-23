#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by any Pipecat LLM service.

Every question about a state goes to the LLM in one out-of-pipeline call,
through the service's ``run_inference()``. The LLM is asked for one JSON
object with an answer per question, and the object is parsed into results.
"""

import json
import re
from collections.abc import Mapping
from typing import Any

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
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
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

DEFAULT_INSTRUCTIONS = (
    "You are a classifier. Each message describes some input and asks one or "
    "more questions about it. Every question has a name and says the shape its "
    "answer must have. Reply with a single JSON object, and nothing else: one "
    "key per question name, each holding that question's answer in the shape "
    "asked for. Probabilities and confidences are numbers from 0 to 1. For "
    'example, a yes or no question named "voicemail" is answered '
    '{"voicemail": {"probability": 0.97}}. A choice question named "department" '
    "with the options billing, support and sales is answered "
    '{"department": {"choice": "billing", "probabilities": {"billing": 0.92, '
    '"support": 0.06, "sales": 0.02}}}. A score question named "mood" on a '
    "scale of three levels is answered with a probability per level, by position, "
    '{"mood": {"probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}}. '
    "Several questions about one input get one object with an answer for each "
    "of them."
)


class LLMClassifier(BaseClassifier):
    """Answers questions by asking an LLM for a JSON object.

    The probabilities are whatever the LLM wrote, so they are not calibrated.
    Any service that implements ``run_inference()`` can back a classifier;
    realtime services cannot. The reply's shape is enforced by the provider
    where the service supports a reply schema, and otherwise asked for in
    the prompt and parsed from the reply.

    Example::

        classifier = LLMClassifier(llm=OpenAILLMService(model="gpt-4o-mini"))
        results = await classifier.yes_no(
            "Hi, you've reached Dana. Leave a message.",
            {"voicemail": YesNoQuestion(instructions="is this a voicemail greeting?")},
        )
        results["voicemail"].probability
    """

    def __init__(
        self,
        *,
        llm: LLMService[Any],
        instructions: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            llm: The LLM that answers the questions.
            instructions: System instructions for the LLM. The default asks
                for one JSON object with an answer per question.
            max_tokens: Cap on the reply's length, for services that take one.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._instructions = instructions or DEFAULT_INSTRUCTIONS
        self._max_tokens = max_tokens

    @property
    def llm(self) -> LLMService[Any]:
        """The LLM that answers the questions."""
        return self._llm

    @property
    def model(self) -> str | None:
        """The LLM service's model."""
        model = self._llm.settings.model
        return model if isinstance(model, str) else None

    async def _ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> tuple[dict[str, ClassifierResult], None]:
        """Answer the questions in one LLM call. The call reports no token usage."""
        context = LLMContext([{"role": "user", "content": self._render(state, questions)}])
        try:
            reply = await self._llm.run_inference(
                context,
                max_tokens=self._max_tokens,
                system_instruction=self._instructions,
                response_schema=self._schema(questions),
            )
        except NotImplementedError as e:
            raise ClassifierError(f"{self._llm} cannot run a one-shot inference") from e
        answers = self._parse(reply or "")
        if len(questions) == 1 and not any(name in answers for name in questions):
            # A lone answer often comes back without its name around it.
            answers = {next(iter(questions)): answers}
        results: dict[str, ClassifierResult] = {}
        for name, question in questions.items():
            answer = answers.get(name)
            if not isinstance(answer, dict):
                raise ClassifierError(f"the LLM gave no answer for {name!r}")
            results[name] = self._result(question, answer)
        return results, None

    def _render(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> str:
        """Write the state and the questions as the message the LLM answers."""
        parts = [f"Input:\n{self._text(state)}"]
        for name, question in questions.items():
            lines = [f'Question "{name}": {self._text(question.instructions)}']
            if isinstance(question, YesNoQuestion):
                if question.yes is not None:
                    lines.append(f"Yes means: {self._text(question.yes)}")
                if question.no is not None:
                    lines.append(f"No means: {self._text(question.no)}")
                lines.append('Answer shape: {"probability": <probability that the answer is yes>}')
            elif isinstance(question, ChoiceQuestion):
                lines.append("Options:")
                for option, description in question.options.items():
                    lines.append(
                        f"- {option}: {self._text(description)}"
                        if description is not None
                        else f"- {option}"
                    )
                lines.append(
                    'Answer shape: {"choice": <the option that fits best>, '
                    '"probabilities": {<a probability per option, summing to 1>}}'
                )
            else:
                lines.append("Scale, lowest first:")
                for index, level in enumerate(question.levels):
                    lines.append(f"{index}: {self._text(level)}")
                positions = ", ".join(f'"{i}": <probability>' for i in range(len(question.levels)))
                lines.append(
                    f'Answer shape: {{"probabilities": {{{positions}}}}}, one per level by '
                    "position, summing to 1"
                )
            parts.append("\n".join(lines))
        names = ", ".join(f'"{name}": <answer to "{name}">' for name in questions)
        parts.append(f"Reply with one JSON object of this shape: {{{names}}}")
        return "\n\n".join(parts)

    def _schema(self, questions: Mapping[str, ClassifierQuestion]) -> dict[str, Any]:
        """The JSON schema of the reply: one answer per question, in its shape."""
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            if isinstance(question, YesNoQuestion):
                answers[name] = self._object({"probability": {"type": "number"}})
            elif isinstance(question, ChoiceQuestion):
                options = list(question.options)
                answers[name] = self._object(
                    {
                        "choice": {"type": "string", "enum": options},
                        "probabilities": self._object({o: {"type": "number"} for o in options}),
                    }
                )
            else:
                positions = [str(i) for i in range(len(question.levels))]
                answers[name] = self._object(
                    {"probabilities": self._object({i: {"type": "number"} for i in positions})}
                )
        return self._object(answers)

    def _object(self, properties: dict[str, Any]) -> dict[str, Any]:
        """A schema for an object with exactly these properties, all required."""
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def _parse(self, reply: str) -> dict[str, Any]:
        """The JSON object in the LLM's reply, fences and prose around it ignored."""
        text = reply.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ClassifierError(f"the LLM did not answer with a JSON object: {reply!r}")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise ClassifierError(f"the LLM's answer is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ClassifierError("the LLM's answer is not a JSON object")
        return data

    def _result(self, question: ClassifierQuestion, answer: dict[str, Any]) -> ClassifierResult:
        """A result built from the LLM's answer to the question."""
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._unit(answer, "probability"))
        if isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(f"the LLM chose {choice!r}, which is not an option")
            given = answer.get("probabilities")
            given = given if isinstance(given, dict) else {}
            probabilities = {o: self._clamp(given.get(o, 0.0)) for o in question.options}
            return ChoiceResult(
                choice=choice, probabilities=probabilities, confidence=probabilities[choice]
            )
        # The LLM gives a probability per level, by position. The score is the
        # probability-weighted position and the confidence the largest share,
        # as with Jev; a distribution that does not sum to 1 is scaled to.
        given = answer.get("probabilities")
        given = given if isinstance(given, dict) else {}
        probabilities = [self._clamp(given.get(str(i), 0.0)) for i in range(len(question.levels))]
        total = sum(probabilities)
        if total <= 0:
            raise ClassifierError("the LLM gave no probability for any level")
        probabilities = [p / total for p in probabilities]
        return ScoreResult(
            score=sum(i * p for i, p in enumerate(probabilities)),
            levels=[
                ScoreLevel(level=level, probability=p)
                for level, p in zip(question.levels, probabilities)
            ],
            confidence=max(probabilities),
        )

    def _text(self, value: Any) -> str:
        """Text for the LLM: a string as is, structured data as JSON."""
        return value if isinstance(value, str) else json.dumps(value, indent=2)

    def _number(self, answer: dict[str, Any], key: str) -> float:
        """The number the LLM wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"the LLM's answer has no usable '{key}'") from e

    def _unit(self, answer: dict[str, Any], key: str) -> float:
        """The number under ``key`` kept between 0 and 1, for probabilities and confidences."""
        return self._clamp(self._number(answer, key))

    def _clamp(self, value: Any) -> float:
        """A number the LLM wrote, kept between 0 and 1."""
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
