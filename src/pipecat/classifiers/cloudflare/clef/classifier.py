#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Clef, Cloudflare's decision model on Workers AI.

:class:`ClefClassifier` turns each question into a request and each reply
into a result, through a :class:`~pipecat.classifiers.cloudflare.clef.client.ClefClient`.
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
from pipecat.classifiers.cloudflare.clef.client import DEFAULT_MODEL, DEFAULT_TIMEOUT, ClefClient
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The most options Clef takes in one choice question.
CLEF_MAX_CHOICE_OPTIONS = 255


class ClefClassifier(BaseClassifier):
    """Answers questions by asking Clef.

    Build one with a Cloudflare account ID and API token to get a client of
    its own, or pass a :class:`ClefClient` to share one between several
    classifiers. A client the classifier created is closed in
    :meth:`cleanup`; a shared one is left to whoever made it.

    Example::

        classifier = ClefClassifier(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            api_key=os.getenv("CLOUDFLARE_API_KEY"),
        )
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        client: ClefClient | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            account_id: Cloudflare account ID, when the classifier should
                have a client of its own.
            api_key: Cloudflare API token, when the classifier should have a
                client of its own.
            client: A client to share. Either ``client``, or both
                ``account_id`` and ``api_key``, are required.
            model: The Clef model a client of its own asks: ``clef``, or
                ``clef-flash`` for faster answers.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not account_id or not api_key:
                raise ValueError("ClefClassifier needs an account ID and API key, or a ClefClient")
            client = ClefClient(
                account_id=account_id, api_key=api_key, model=model, timeout=timeout
            )
        self._client = client

    @property
    def client(self) -> ClefClient:
        """The client this classifier asks through."""
        return self._client

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to Workers AI ahead of the first question.

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
        """The Clef model the questions go to."""
        return self._client.model

    async def _ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        answers, usage = await self._client.ask(
            state, {name: self._to_clef(q) for name, q in questions.items()}
        )
        results = {
            name: self._from_clef(question, answers[name]) for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    def _to_clef(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in Clef's own format."""
        if isinstance(question, YesNoQuestion):
            clef: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.yes is not None or question.no is not None:
                clef["criteria"] = {"true": question.yes or "", "false": question.no or ""}
            return clef
        if isinstance(question, ChoiceQuestion):
            if len(question.options) > CLEF_MAX_CHOICE_OPTIONS:
                raise ClassifierError(
                    f"Clef takes at most {CLEF_MAX_CHOICE_OPTIONS} options, "
                    f"got {len(question.options)}"
                )
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.options,
            }
        return {"type": "score", "instructions": question.instructions, "criteria": question.levels}

    def _from_clef(self, question: ClassifierQuestion, answer: dict[str, Any]) -> ClassifierResult:
        """A result built from Clef's answer to the question."""
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._number(answer, "noul"))
        if isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(f"Clef chose {choice!r}, which is not an option")
            probabilities = self._probabilities(answer, list(question.options))
            return ChoiceResult(
                choice=choice,
                probabilities=probabilities,
                confidence=self._number(answer, "confidence"),
            )
        # Clef keys level probabilities by position.
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
        """The number Clef wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"Clef reply has no usable '{key}'") from e

    def _probabilities(self, answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
        """The probability Clef wrote for each key, 0 for the ones it left out."""
        given = answer.get("probabilities") or {}
        try:
            return {key: float(given.get(key, 0.0)) for key in keys}
        except (TypeError, ValueError) as e:
            raise ClassifierError(f"Clef reply has an unusable probability: {e}") from e
