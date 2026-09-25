#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Jev, TypeSafe's hosted classification model.

:class:`JevClassifier` turns each question into a request and each reply
into a result, through a :class:`~pipecat.classifiers.jev.client.JevClient`.
"""

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
from pipecat.classifiers.jev.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    JevClient,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The most options Jev takes in one choice question.
JEV_MAX_CHOICE_OPTIONS = 255


class JevClassifier(BaseClassifier):
    """Answers questions by asking Jev.

    Jev's probabilities are calibrated. Build one with an API key to get a
    client of its own, or pass a :class:`JevClient` to share one between
    several classifiers. A client the classifier created is closed in
    :meth:`cleanup`; a shared one is left to whoever made it.

    Example::

        client = JevClient(api_key=os.getenv("TYPESAFE_API_KEY"))
        turn_classifier = JevClassifier(client=client)
        voicemail_classifier = JevClassifier(client=client)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: JevClient | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: Jev API key, when the classifier should have a client
                of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
            model: The Jev model a client of its own asks.
            base_url: Where a client of its own sends its questions.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key:
                raise ValueError("JevClassifier needs an API key or a JevClient")
            client = JevClient(api_key=api_key, model=model, base_url=base_url, timeout=timeout)
        self._client = client

    @property
    def client(self) -> JevClient:
        """The client this classifier asks through."""
        return self._client

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to Jev ahead of the first question.

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
        """The Jev model the questions go to."""
        return self._client.model

    async def _ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        answers, usage = await self._client.ask(
            state, {name: self._to_jev(q) for name, q in questions.items()}
        )
        results = {
            name: self._from_jev(question, answers[name]) for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    def _to_jev(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in Jev's own format."""
        if isinstance(question, YesNoQuestion):
            jev: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.yes is not None or question.no is not None:
                jev["criteria"] = {"true": question.yes or "", "false": question.no or ""}
            return jev
        if isinstance(question, ChoiceQuestion):
            if len(question.options) > JEV_MAX_CHOICE_OPTIONS:
                raise ClassifierError(
                    f"Jev takes at most {JEV_MAX_CHOICE_OPTIONS} options, got {len(question.options)}"
                )
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.options,
            }
        return {"type": "score", "instructions": question.instructions, "criteria": question.levels}

    def _from_jev(self, question: ClassifierQuestion, answer: dict[str, Any]) -> ClassifierResult:
        """A result built from Jev's answer to the question."""
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._number(answer, "noul"))
        if isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(f"Jev chose {choice!r}, which is not an option")
            probabilities = self._probabilities(answer, list(question.options))
            return ChoiceResult(
                choice=choice,
                probabilities=probabilities,
                confidence=self._number(answer, "confidence"),
            )
        # Jev keys level probabilities by position.
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

    def _number(self, answer: dict[str, Any], key: str) -> float:
        """The number Jev wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"Jev reply has no usable '{key}'") from e

    def _probabilities(self, answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
        """The probability Jev wrote for each key, 0 for the ones it left out."""
        given = answer.get("probabilities") or {}
        try:
            return {key: float(given.get(key, 0.0)) for key in keys}
        except (TypeError, ValueError) as e:
            raise ClassifierError(f"Jev reply has an unusable probability: {e}") from e
