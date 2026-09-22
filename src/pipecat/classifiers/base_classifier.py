#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifiers answer typed questions about some state.

A classifier is a plain object. Whoever needs answers creates one, keeps it,
and calls it. It does not sit in a pipeline and no frames flow into it. A
question is one of :class:`YesNoQuestion`, :class:`ChoiceQuestion` or
:class:`ScoreQuestion`. Questions are asked by name, several about one state
at once: :meth:`BaseClassifier.ask` takes any mix of kinds, and
:meth:`BaseClassifier.yes_no`, :meth:`BaseClassifier.choice` and
:meth:`BaseClassifier.score` take questions of one kind and return typed
results.
"""

import json
import time
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, TypeAlias, TypeVar

from pydantic import BaseModel, Field

from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
)
from pipecat.utils.base_object import BaseObject
from pipecat.workers.base_worker import BaseWorker


class ClassifierError(Exception):
    """A classifier could not answer a question."""

    pass


class YesNoQuestion(BaseModel):
    """Whether the state satisfies a criteria.

    Parameters:
        instructions: What is being checked for, as a yes or no question.
            Text, or structured data holding the question in one field and
            what it refers to in others.
        yes: What counts as a yes, when the question alone leaves it open.
        no: What counts as a no.
    """

    instructions: str | dict[str, Any] | list[Any]
    yes: str | dict[str, Any] | list[Any] | None = None
    no: str | dict[str, Any] | list[Any] | None = None


class ChoiceQuestion(BaseModel):
    """Which of several options fits the state.

    Parameters:
        instructions: What is being decided.
        options: The options to choose from, each mapped to a description
            of when it applies, or ``None`` when the option itself says
            enough.
    """

    instructions: str | dict[str, Any] | list[Any]
    options: dict[str, str | dict[str, Any] | list[Any] | None]


class ScoreQuestion(BaseModel):
    """Where the state falls on an ordered scale.

    Parameters:
        instructions: What is being rated.
        levels: The levels of the scale in order, lowest first, each described in a few
            words or as structured data. At least two.
    """

    instructions: str | dict[str, Any] | list[Any]
    levels: list[str | dict[str, Any] | list[Any]] = Field(min_length=2)


ClassifierQuestion: TypeAlias = YesNoQuestion | ChoiceQuestion | ScoreQuestion


class YesNoResult(BaseModel):
    """Answer to a :class:`YesNoQuestion`.

    Parameters:
        probability: How likely the answer is yes, from 0 to 1.
    """

    probability: float

    @property
    def is_yes(self) -> bool:
        """Whether yes is the likelier answer.

        Callers that need more certainty than that compare ``probability``
        with a threshold of their own.
        """
        return self.probability >= 0.5


class ChoiceResult(BaseModel):
    """Answer to a :class:`ChoiceQuestion`.

    Parameters:
        choice: The option that fits best.
        probabilities: How likely each option is, keyed by option.
        confidence: How sure the classifier is of ``choice``, from 0 to 1.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreResult(BaseModel):
    """Answer to a :class:`ScoreQuestion`.

    Parameters:
        score: Where the state falls on the scale, as a position from 0 (the
            first level) to one less than the number of levels. It may fall
            between two levels.
        probabilities: How likely each level is, keyed by level description
            (structured levels are keyed by their JSON text).
        confidence: How sure the classifier is of ``score``, from 0 to 1.
    """

    score: float
    probabilities: dict[str, float]
    confidence: float

    @staticmethod
    def level_key(level: Any) -> str:
        """The key a level gets in ``probabilities``.

        Args:
            level: A level of the scale, text or structured data.

        Returns:
            The level itself when it is text, otherwise its JSON text.
        """
        return level if isinstance(level, str) else json.dumps(level, sort_keys=True)


ClassifierResult: TypeAlias = YesNoResult | ChoiceResult | ScoreResult

R = TypeVar("R", YesNoResult, ChoiceResult, ScoreResult)


class BaseClassifier(BaseObject):
    """Answers typed questions about a state.

    Every question is about a ``state``: plain text, or structured data such
    as a transcript with speaker labels or a trimmed screen snapshot.
    :meth:`ask` answers any number of questions about one state, by name;
    the three typed methods are built on it and take questions of one kind.
    Subclasses implement :meth:`_ask`.

    The owner calls :meth:`setup` once before the first question and
    :meth:`cleanup` once when it is done.

    Event handlers available:

    - on_metrics: Called after every call with its metrics, the time it
      took and, when the classifier knows it, the tokens it used. A
      classifier cannot push frames, so the owner is the one to put them in
      a :class:`~pipecat.frames.frames.MetricsFrame`.

    Example::

        @classifier.event_handler("on_metrics")
        async def on_metrics(classifier, data: list[MetricsData]):
            await processor.push_frame(MetricsFrame(data=data))
    """

    def __init__(self, **kwargs):
        """Initialize the classifier.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self._register_event_handler("on_metrics")

    @property
    def model_name(self) -> str | None:
        """The model that answers, named in the metrics."""
        return None

    async def setup(self, worker: BaseWorker):
        """Prepare the classifier to answer questions.

        Args:
            worker: The worker the owner runs in. Its task manager runs the
                classifier's tasks and event handlers, and implementations
                that need the worker itself keep it.
        """
        await super().setup(worker.task_manager)

    async def ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> dict[str, ClassifierResult]:
        """Answer several questions about one state.

        Args:
            state: What the questions are about.
            questions: The questions, by name.

        Returns:
            One result per question, by the same names, each of the type
            its question calls for.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        started = time.perf_counter()
        results, usage = await self._ask(state, questions)
        await self._call_event_handler(
            "on_metrics", self._metrics(time.perf_counter() - started, usage)
        )
        return results

    async def yes_no(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, YesNoQuestion]
    ) -> dict[str, YesNoResult]:
        """Ask whether the state satisfies each criteria.

        Args:
            state: What the questions are about.
            questions: The questions, by name.

        Returns:
            How likely each answer is yes, by the same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        return self._typed(await self.ask(state, questions), YesNoResult)

    async def choice(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ChoiceQuestion]
    ) -> dict[str, ChoiceResult]:
        """Ask which option fits the state, for each question.

        Args:
            state: What the questions are about.
            questions: The questions, by name.

        Returns:
            The option that fits and how likely each one is, by the same
            names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        return self._typed(await self.ask(state, questions), ChoiceResult)

    async def score(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ScoreQuestion]
    ) -> dict[str, ScoreResult]:
        """Ask where the state falls on each scale.

        Args:
            state: What the questions are about.
            questions: The questions, by name.

        Returns:
            The position on each scale and how likely each level is, by the
            same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        return self._typed(await self.ask(state, questions), ScoreResult)

    @abstractmethod
    async def _ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, ClassifierQuestion]
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions, and say what tokens the call used if that is known."""
        pass

    def _metrics(self, seconds: float, usage: LLMTokenUsage | None) -> list[MetricsData]:
        data: list[MetricsData] = [
            ProcessingMetricsData(processor=self.name, model=self.model_name, value=seconds)
        ]
        if usage is not None:
            data.append(
                LLMUsageMetricsData(processor=self.name, model=self.model_name, value=usage)
            )
        return data

    def _typed(self, results: dict[str, ClassifierResult], result_type: type[R]) -> dict[str, R]:
        typed: dict[str, R] = {}
        for name, result in results.items():
            if not isinstance(result, result_type):
                raise ClassifierError(
                    f"expected a {result_type.__name__} for {name!r}, got {type(result).__name__}"
                )
            typed[name] = result
        return typed
