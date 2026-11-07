#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Hume's Audio Decisions API, which answers questions about audio.

:class:`HumeAudioClassifier` turns each question into a request and
each reply into a result, through a
:class:`~pipecat.classifiers.hume.audio_decisions.client.HumeAudioDecisionsClient`.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from pipecat.audio.utils import pcm_to_wav
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierAudio,
    ClassifierError,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    ScoreLevel,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.hume.audio_decisions.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    HumeAudioDecisionsClient,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The longest context Hume reads, in characters.
HUME_MAX_CONTEXT = 1000


class HumeAudioClassifier(BaseClassifier):
    """Answers questions about a clip of audio by asking Hume's Audio Decisions API.

    Hume hears the clip, 5 seconds to 5 minutes of it, and reads the state as
    the clip's ``context``, such as who is speaking and what they said. Every
    call needs exactly one clip in its ``inputs``: any other number is
    refused with a :class:`~pipecat.classifiers.base_classifier.ClassifierError`. A
    state longer than Hume reads keeps its last 1000 characters. Its
    probabilities estimate the share of human raters who would give each
    answer. Clips are sent as 16 kHz mono WAV, which is how the model hears
    them.

    Build one with an API key to get a client of its own, or pass a
    :class:`HumeAudioDecisionsClient` to share one between several
    classifiers. A client the classifier created is closed in
    :meth:`cleanup`; a shared one is left to whoever made it.

    Example::

        classifier = HumeAudioClassifier(api_key=os.getenv("HUME_API_KEY"))
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: HumeAudioDecisionsClient | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: Hume API key, when the classifier should have a client
                of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
            base_url: Where a client of its own sends its questions.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key:
                raise ValueError("HumeAudioClassifier needs an API key or a client")
            client = HumeAudioDecisionsClient(api_key=api_key, base_url=base_url, timeout=timeout)
        self._client = client

    @property
    def client(self) -> HumeAudioDecisionsClient:
        """The client this classifier asks through."""
        return self._client

    @property
    def supports_audio(self) -> bool:
        """Hume hears audio."""
        return True

    @property
    def min_audio_secs(self) -> float:
        """Hume hears at least 5 seconds of audio."""
        return 5.0

    @property
    def max_audio_secs(self) -> float:
        """Hume hears at most 5 minutes of audio."""
        return 300.0

    @property
    def audio_sample_rate(self) -> int:
        """Hume's model hears audio at 16 kHz."""
        return 16000

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to Hume ahead of the first question.

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

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions about the clip in one request."""
        clips = [item for item in inputs if isinstance(item, ClassifierAudio)]
        if len(clips) != 1:
            raise ClassifierError(
                f"{self} answers questions about one clip of audio, got {len(clips)}"
            )
        audio = clips[0]
        answers, usage = await self._client.ask(
            pcm_to_wav(audio.audio, audio.sample_rate),
            {name: self._to_hume(question) for name, question in questions.items()},
            self._context(state),
        )
        results = {
            name: self._from_hume(question, answers[name]) for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=0,
            total_tokens=usage.input_tokens,
        )

    def _context(self, state: str | dict[str, Any] | list[Any]) -> str | None:
        """The state as Hume's context: its text, or its JSON, and its last characters only."""
        if not state:
            return None
        text = state if isinstance(state, str) else json.dumps(state)
        return text[-HUME_MAX_CONTEXT:]

    def _text(self, value: str | dict[str, Any] | list[Any]) -> str:
        """Text, or structured data as its JSON, for a field Hume reads as text."""
        return value if isinstance(value, str) else json.dumps(value)

    def _to_hume(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in Hume's own format."""
        hume: dict[str, Any] = {"instructions": self._text(question.instructions)}
        if isinstance(question, YesNoQuestion):
            hume["type"] = "noul"
            criteria = {
                key: self._text(value)
                for key, value in (("true", question.yes), ("false", question.no))
                if value is not None
            }
            if criteria:
                hume["criteria"] = criteria
        elif isinstance(question, ChoiceQuestion):
            hume["type"] = "choice"
            hume["criteria"] = [
                self._option(name, description) for name, description in question.options.items()
            ]
        else:
            hume["type"] = "score"
            hume["criteria"] = [self._text(level) for level in question.levels]
        return hume

    def _option(
        self, name: str, description: str | dict[str, Any] | list[Any] | None
    ) -> dict[str, str]:
        """A choice option in Hume's own format."""
        option = {"name": name}
        if description is not None:
            option["description"] = self._text(description)
        return option

    def _from_hume(self, question: ClassifierQuestion, answer: dict[str, Any]) -> ClassifierResult:
        """A result built from Hume's answer to the question."""
        if isinstance(question, YesNoQuestion):
            result: ClassifierResult = YesNoResult(probability=self._number(answer, "noul"))
        elif isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(f"Hume chose {choice!r}, which is not an option")
            result = ChoiceResult(
                choice=choice,
                probabilities=self._probabilities(answer, list(question.options)),
                confidence=self._number(answer, "confidence"),
            )
        else:
            # Hume keys level probabilities by position.
            positions = [str(index) for index in range(len(question.levels))]
            probabilities = self._probabilities(answer, positions)
            result = ScoreResult(
                score=self._number(answer, "score"),
                levels=[
                    ScoreLevel(level=level, probability=probabilities[position])
                    for level, position in zip(question.levels, positions)
                ],
                confidence=self._number(answer, "confidence"),
            )
        return result

    def _number(self, answer: dict[str, Any], key: str) -> float:
        """The number Hume wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"Hume reply has no usable '{key}'") from e

    def _probabilities(self, answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
        """The probability Hume wrote for each key, 0 for the ones it left out."""
        given = answer.get("probabilities") or {}
        try:
            return {key: float(given.get(key, 0.0)) for key in keys}
        except (TypeError, ValueError, AttributeError) as e:
            raise ClassifierError(f"Hume reply has an unusable probability: {e}") from e
