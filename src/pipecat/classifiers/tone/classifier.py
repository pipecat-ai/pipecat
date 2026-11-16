#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Pipecat Tone's reader, a local model of how one turn of text was meant.

Pipecat Tone (https://huggingface.co/pipecat-ai/tone-models) reads one turn
of text, with the turn before it, and returns the speaker's emotion, whether
the turn is sarcastic, the move it makes and whether it's only a listening
sound, each with a calibrated confidence. It runs on the CPU in a few
milliseconds.

**Alpha.** These types are likely to change between releases.
"""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.tone.model import TONE_REPO, TONE_REVISION, PipecatToneReaderModel
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.models import load_shared_model


class PipecatToneClassifier(BaseClassifier):
    """Answers Pipecat Tone's own questions about one turn of text.

    It answers only the questions built into Tone's model, the ones in
    :attr:`questions`: the turn's ``emotion`` and ``move``, whether it's
    sarcastic (``sarcasm``), and whether it's only a listening sound such as
    "mhm" (``backchannel``). Any other question raises a
    :class:`~pipecat.classifiers.base_classifier.ClassifierError`. The
    questions take their options from the model, which :meth:`setup` loads.

    The state is the turn's text, or ``{"user": ..., "assistant": ...}`` to
    read it with the bot's words before it.

    The model loads in :meth:`setup`, once a process, and every classifier
    shares it. The first load downloads it from Hugging Face.

    Example::

        classifier = PipecatToneClassifier()
        await classifier.setup(task_manager)

        results = await classifier.ask(
            {"assistant": "Your next visit is in forty minutes.", "user": "Oh great."},
            classifier.questions,
        )
        sarcastic = results["sarcasm"].probability
    """

    def __init__(
        self,
        *,
        revision: str = TONE_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            revision: The model's release on Hugging Face, such as ``"reader-rc2"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as a release's ``reader``. ``revision`` is then unused.
            precision: The weights to load: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._revision = revision
        self._model_dir = Path(model_dir) if model_dir else None
        self._precision: Literal["fp16", "fp32"] = precision
        self._threads = threads
        self._model: PipecatToneReaderModel | None = None
        self._questions: dict[str, ClassifierQuestion] = {}

    @property
    def model(self) -> str:
        """The model's folder, or its release on Hugging Face."""
        return str(self._model_dir) if self._model_dir else f"{TONE_REPO}@{self._revision}"

    @property
    def questions(self) -> dict[str, ClassifierQuestion]:
        """The questions the model answers, by name.

        Raises:
            ClassifierError: If the model isn't loaded yet.
        """
        if not self._model:
            raise ClassifierError(f"{self}: the questions come from the model, loaded in setup()")
        return dict(self._questions)

    async def setup(self, task_manager: BaseTaskManager):
        """Load the model, or share the one already loaded.

        Args:
            task_manager: The task manager of the owner.

        Raises:
            ClassifierError: If the model can't be loaded.
        """
        await super().setup(task_manager)
        key = ("reader", self._model_dir or self._revision, self._precision, self._threads)
        try:
            model = await load_shared_model(key, self._open_model)
            self._questions = self._model_questions(model)
        except Exception as e:
            raise ClassifierError(f"couldn't load Pipecat Tone: {e!r}") from e
        self._model = model

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions with one read of the turn."""
        if not self._model:
            raise ClassifierError(f"{self}: the model isn't loaded, see setup()")
        heads = {name: self._head(question) for name, question in questions.items()}
        text, previous = self._turn(state)
        tags = await self._tag(text, previous)
        return {name: self._result(head, tags) for name, head in heads.items()}, None

    def _open_model(self) -> PipecatToneReaderModel:
        """The model from its folder, or downloaded from Hugging Face."""
        if self._model_dir:
            return PipecatToneReaderModel(
                self._model_dir, precision=self._precision, threads=self._threads
            )
        return PipecatToneReaderModel.download(
            revision=self._revision, precision=self._precision, threads=self._threads
        )

    def _model_questions(self, model: PipecatToneReaderModel) -> dict[str, ClassifierQuestion]:
        """The questions the model answers, with its own options."""
        return {
            "emotion": ChoiceQuestion(
                instructions="Which feeling the turn's words show.",
                options={emotion: None for emotion in model.emotions},
            ),
            "move": ChoiceQuestion(
                instructions="What the turn does in the conversation.",
                options={move: None for move in model.moves},
            ),
            "sarcasm": YesNoQuestion(instructions="Is the turn sarcastic?"),
            "backchannel": YesNoQuestion(
                instructions=(
                    'Is the turn only a listening sound, such as "mhm", that acknowledges '
                    "the speaker without taking the turn?"
                )
            ),
        }

    def _head(self, question: ClassifierQuestion) -> str:
        """Which of the model's questions this is, by name, or a ClassifierError if none."""
        head = next((name for name, own in self._questions.items() if own == question), None)
        if head is None:
            raise ClassifierError(
                f"{self} answers only its own questions, from PipecatToneClassifier.questions"
            )
        return head

    def _turn(self, state: str | dict[str, Any] | list[Any]) -> tuple[str, str | None]:
        """The turn's text and the bot's words before it."""
        if isinstance(state, str):
            text, previous = state, None
        elif (
            isinstance(state, dict)
            and isinstance(state.get("user"), str)
            and isinstance(state.get("assistant", ""), str | None)
        ):
            text, previous = state["user"], state.get("assistant")
        else:
            raise ClassifierError(
                f"{self} reads a turn: text, or a dict with 'user' and optionally 'assistant'"
            )
        return text, previous

    async def _tag(self, text: str, previous: str | None) -> dict[str, Any]:
        """The model's tags for the turn, read off the event loop."""
        assert self._model
        try:
            return await asyncio.to_thread(self._model.tag, text, previous)
        except Exception as e:
            raise ClassifierError(f"couldn't read the turn: {e!r}") from e

    def _result(self, head: str, tags: dict[str, Any]) -> ClassifierResult:
        """The answer to one of the model's questions, from its tags."""
        tag = tags[head]
        if isinstance(self._questions[head], ChoiceQuestion):
            result: ClassifierResult = ChoiceResult(
                choice=tag["label"],
                probabilities=tag["probabilities"],
                confidence=tag["confidence"],
            )
        else:
            result = YesNoResult(probability=tag["probability"])
        return result
