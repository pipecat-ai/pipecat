#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by OpenAI's Decisions API.

:class:`OpenAIDecisionsClassifier` turns each question into a request and
each reply into a result, through a
:class:`~pipecat.classifiers.openai.decisions.client.OpenAIDecisionsClient`.
"""

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger

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
from pipecat.classifiers.openai.decisions.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OpenAIDecisionsClient,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The most options the Decisions API takes in one choice question.
OPENAI_DECISIONS_MAX_CHOICE_OPTIONS = 255


class OpenAIDecisionsClassifier(BaseClassifier):
    """Answers questions by asking OpenAI's Decisions API.

    The API takes only text, so structured state, instructions and
    descriptions are sent as JSON. Build one with an API key to get a client
    of its own, or pass an :class:`OpenAIDecisionsClient` to share one
    between several classifiers. A client the classifier created is closed
    in :meth:`cleanup`; a shared one is left to whoever made it.

    Example::

        classifier = OpenAIDecisionsClassifier(api_key=os.getenv("OPENAI_API_KEY"))
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: OpenAIDecisionsClient | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: OpenAI API key, when the classifier should have a client
                of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
            model: The decision model a client of its own asks.
            base_url: Where a client of its own sends its questions.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key:
                raise ValueError(
                    "OpenAIDecisionsClassifier needs an API key or an OpenAIDecisionsClient"
                )
            client = OpenAIDecisionsClient(
                api_key=api_key, model=model, base_url=base_url, timeout=timeout
            )
        self._client = client

    @property
    def client(self) -> OpenAIDecisionsClient:
        """The client this classifier asks through."""
        return self._client

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to OpenAI ahead of the first question.

        A connection that cannot be opened now is only a warning: the first
        question opens it itself.

        Args:
            task_manager: The task manager of the owner.
        """
        await super().setup(task_manager)
        try:
            await self._client.connect()
        except ClassifierError as e:
            logger.warning(f"{self}: {e}")

    async def cleanup(self):
        """Close the client if this classifier created it."""
        await super().cleanup()
        if self._owns_client:
            await self._client.close()

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._client.model

    async def _ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        answers, usage = await self._client.ask(
            self._text(state), {name: self._to_openai(q) for name, q in questions.items()}
        )
        results = {
            name: self._from_openai(name, question, answers[name])
            for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    def _to_openai(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in the Decisions API's own format."""
        if isinstance(question, YesNoQuestion):
            # A predicate has no field for what counts as each answer, so it
            # goes in the instructions.
            lines = [self._text(question.instructions)]
            if question.yes is not None:
                lines.append(f"Yes means: {self._text(question.yes)}")
            if question.no is not None:
                lines.append(f"No means: {self._text(question.no)}")
            return {"type": "predicate", "instructions": "\n".join(lines)}
        if isinstance(question, ChoiceQuestion):
            if len(question.options) > OPENAI_DECISIONS_MAX_CHOICE_OPTIONS:
                raise ClassifierError(
                    f"OpenAI Decisions takes at most {OPENAI_DECISIONS_MAX_CHOICE_OPTIONS} "
                    f"options, got {len(question.options)}"
                )
            return {
                "type": "choice",
                "instructions": self._text(question.instructions),
                "choices": [
                    {"value": option}
                    if description is None
                    else {"value": option, "description": self._text(description)}
                    for option, description in question.options.items()
                ],
            }
        return {
            "type": "score",
            "instructions": self._text(question.instructions),
            "levels": [{"label": self._text(level)} for level in question.levels],
        }

    def _from_openai(
        self, name: str, question: ClassifierQuestion, answer: dict[str, Any]
    ) -> ClassifierResult:
        """A result built from the Decisions API's answer to the question."""
        if answer.get("type") == "refusal":
            raise ClassifierError(f"OpenAI Decisions refused to answer {name!r}")
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._number(answer, "probability"))
        if isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(f"OpenAI Decisions chose {choice!r}, which is not an option")
            probabilities = self._probabilities(answer, list(question.options))
            return ChoiceResult(
                choice=choice,
                probabilities=probabilities,
                confidence=self._number(answer, "confidence"),
            )
        # The Decisions API keys level probabilities by position.
        positions = [str(index) for index in range(len(question.levels))]
        probabilities = self._probabilities(answer, positions)
        return ScoreResult(
            score=self._number(answer, "score"),
            levels=[
                ScoreLevel(level=level, probability=probabilities[position])
                for level, position in zip(question.levels, positions)
            ],
            confidence=self._number(answer, "confidence"),
        )

    def _text(self, value: Any) -> str:
        """Text for the Decisions API: a string as is, structured data as JSON."""
        return value if isinstance(value, str) else json.dumps(value, indent=2)

    def _number(self, answer: dict[str, Any], key: str) -> float:
        """The number the API wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"OpenAI Decisions reply has no usable '{key}'") from e

    def _probabilities(self, answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
        """The probability the API gave for each key, 0 for the ones it left out.

        The API lists probabilities as ``{"value", "probability"}`` items,
        where the value is a choice's option or a level's position.
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
            raise ClassifierError(f"OpenAI Decisions reply has an unusable probability: {e}") from e
