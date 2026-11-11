#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Pipecat Voice Reader, a local model of how a clip of speech sounds.

Pipecat Voice Reader (https://huggingface.co/pipecat-ai/voice-reader) hears
up to the last 8 seconds of a speaker's turn and returns their tone of voice,
how sarcastic, sad and upset they sound, whether they laugh, how activated the
voice is and how fast they speak. It runs on the CPU in roughly a tenth of a
second.

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
    ClassifierAudio,
    ClassifierError,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.voice.model import (
    VOICE_REPO,
    VOICE_REVISION,
    VOICE_SAMPLE_RATE,
    VOICE_WINDOW_SECS,
    PipecatVoiceReaderModel,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.models import load_shared_model

#: The top of the ``speech_rate`` scale, in syllables per second.
_MAX_SPEECH_RATE = 12

#: The model's questions, each named after the output that answers it.
_QUESTIONS: dict[str, ClassifierQuestion] = {
    "tone": ChoiceQuestion(
        instructions="How does the voice sound?",
        options={
            "positive": "warm, bright, pleased",
            "neutral": None,
            "sad": "low, deflated",
            "frustrated": "tense, irritated, clipped",
        },
    ),
    "sarcasm": YesNoQuestion(instructions="Would a listener hear the turn as sarcastic?"),
    "laughter": YesNoQuestion(instructions="Is there laughter in the turn?"),
    "sad": YesNoQuestion(instructions="Would a listener hear sadness in the turn?"),
    "arousal": ScoreQuestion(
        instructions="How activated does the voice sound?",
        levels=["flat, low energy", "highly animated"],
    ),
    "upset": ScoreQuestion(
        instructions="How clearly is the speaker upset?",
        levels=["not at all", "unmistakably"],
    ),
    "speech_rate": ScoreQuestion(
        instructions="How fast is the speech, in syllables per second?",
        levels=[f"{rate} syllables per second" for rate in range(_MAX_SPEECH_RATE + 1)],
    ),
}


class PipecatVoiceClassifier(BaseClassifier):
    """Answers Pipecat Voice Reader's own questions about a clip of speech.

    It answers only the questions built into the model, the ones in
    :attr:`questions`: the speaker's ``tone`` of voice, the chance a listener
    would hear the turn as sarcastic (``sarcasm``), hear ``laughter`` in it or
    hear sadness (``sad``), how activated the voice is (``arousal``), how
    clearly the speaker is ``upset``, and their ``speech_rate`` in syllables
    per second. Any other question raises a
    :class:`~pipecat.classifiers.base_classifier.ClassifierError`. A question
    is known by what it asks, so it can be asked under any name.

    Every call needs exactly one clip in its ``inputs``, from 0.8 to 8 seconds
    long: any other number is refused with a
    :class:`~pipecat.classifiers.base_classifier.ClassifierError`. The model
    hears the audio alone, so the state isn't read. One read of the clip
    answers every question asked.

    The model's ``sarcasm`` rarely passes 0.5, so
    :attr:`~pipecat.classifiers.base_classifier.YesNoResult.is_yes` is seldom
    true of it: compare its probability with :attr:`sarcasm_threshold`
    instead.

    The model loads in :meth:`setup`, once a process, and every classifier of
    the same variant shares it. The first load downloads it from Hugging Face.

    Example::

        classifier = PipecatVoiceClassifier()
        await classifier.setup(task_manager)

        results = await classifier.ask(
            {},
            classifier.questions,
            inputs=[ClassifierAudio(audio=pcm, sample_rate=16000)],
        )
        tone = results["tone"].choice
    """

    def __init__(
        self,
        *,
        variant: Literal["base", "tiny"] = "base",
        revision: str = VOICE_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            variant: The model's size: ``"base"`` reads tone and sadness best,
                and ``"tiny"`` costs about half the CPU.
            revision: The model's release on Hugging Face, such as ``"rc6"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as one of a release's variants. ``variant`` and
                ``revision`` are then unused.
            precision: The weights to load: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._variant: Literal["base", "tiny"] = variant
        self._revision = revision
        self._model_dir = Path(model_dir) if model_dir else None
        self._precision: Literal["int8", "fp32"] = precision
        self._threads = threads
        self._model: PipecatVoiceReaderModel | None = None

    @property
    def model(self) -> str:
        """The model's folder, or its release and variant on Hugging Face."""
        if self._model_dir:
            return str(self._model_dir)
        return f"{VOICE_REPO}@{self._revision}/{self._variant}"

    @property
    def questions(self) -> dict[str, ClassifierQuestion]:
        """The questions the model answers, by name."""
        return dict(_QUESTIONS)

    @property
    def sarcasm_threshold(self) -> float:
        """The ``sarcasm`` probability from which the model's release counts a turn as sarcastic.

        Raises:
            ClassifierError: If the model isn't loaded yet.
        """
        if not self._model:
            raise ClassifierError(f"{self}: the threshold comes from the model, loaded in setup()")
        return self._model.sarcasm_threshold

    @property
    def supports_audio(self) -> bool:
        """The model hears audio."""
        return True

    @property
    def min_audio_secs(self) -> float:
        """The model's readings mean little under 0.8 seconds of audio."""
        return 0.8

    @property
    def max_audio_secs(self) -> float:
        """The model hears at most 8 seconds of audio."""
        return VOICE_WINDOW_SECS

    @property
    def audio_sample_rate(self) -> int:
        """The model hears audio at 16 kHz."""
        return VOICE_SAMPLE_RATE

    async def setup(self, task_manager: BaseTaskManager):
        """Load the model, or share the one already loaded.

        Args:
            task_manager: The task manager of the owner.

        Raises:
            ClassifierError: If the model can't be loaded, or its tones aren't
                the options of the ``tone`` question.
        """
        await super().setup(task_manager)
        key = (
            "voice-reader",
            self._model_dir or (self._revision, self._variant),
            self._precision,
            self._threads,
        )
        try:
            model = await load_shared_model(key, self._open_model)
        except Exception as e:
            raise ClassifierError(f"couldn't load Pipecat Voice Reader: {e!r}") from e
        options = list(self._tone_question().options)
        if model.tones != options:
            raise ClassifierError(
                f"{self}: the model's tones are {model.tones}, and the tone question's {options}"
            )
        self._model = model

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions with one read of the clip."""
        if not self._model:
            raise ClassifierError(f"{self}: the model isn't loaded, see setup()")
        clips = [item for item in inputs if isinstance(item, ClassifierAudio)]
        if len(clips) != 1:
            raise ClassifierError(
                f"{self} answers questions about one clip of audio, got {len(clips)}"
            )
        heads = {name: self._head(question) for name, question in questions.items()}
        reading = await self._read(clips[0])
        return {name: self._result(head, reading) for name, head in heads.items()}, None

    def _open_model(self) -> PipecatVoiceReaderModel:
        """The model from its folder, or downloaded from Hugging Face."""
        if self._model_dir:
            return PipecatVoiceReaderModel(
                self._model_dir, precision=self._precision, threads=self._threads
            )
        return PipecatVoiceReaderModel.download(
            revision=self._revision,
            variant=self._variant,
            precision=self._precision,
            threads=self._threads,
        )

    def _tone_question(self) -> ChoiceQuestion:
        """The ``tone`` question, whose options are the model's tones."""
        question = _QUESTIONS["tone"]
        assert isinstance(question, ChoiceQuestion)
        return question

    def _head(self, question: ClassifierQuestion) -> str:
        """Which of the model's outputs answers the question, or a ClassifierError if none."""
        assert self._model
        head = next((name for name, own in _QUESTIONS.items() if own == question), None)
        if head is None:
            raise ClassifierError(
                f"{self} answers only its own questions, from PipecatVoiceClassifier.questions"
            )
        if head not in self._model.outputs:
            raise ClassifierError(f"{self}: the model has no {head!r} output")
        return head

    async def _read(self, clip: ClassifierAudio) -> dict[str, Any]:
        """The model's values for the clip, read off the event loop."""
        assert self._model
        try:
            return await asyncio.to_thread(self._model.read, clip.audio)
        except Exception as e:
            raise ClassifierError(f"couldn't read the clip: {e!r}") from e

    def _result(self, head: str, reading: dict[str, Any]) -> ClassifierResult:
        """The answer to one of the model's questions, from its values for the clip."""
        question, value = _QUESTIONS[head], reading[head]
        if isinstance(question, ChoiceQuestion):
            choice = max(value, key=value.get)
            result: ClassifierResult = ChoiceResult(
                choice=choice, probabilities=value, confidence=value[choice]
            )
        elif isinstance(question, YesNoQuestion):
            result = YesNoResult(probability=value)
        else:
            result = self._score(question, value)
        return result

    def _score(self, question: ScoreQuestion, value: float) -> ScoreResult:
        """A value as a position on the question's scale, between the two levels around it.

        ``arousal`` and ``upset`` run from 0 to 1 over two levels, and
        ``speech_rate`` is in syllables per second, where each level is one more.
        """
        score = min(max(value, 0.0), len(question.levels) - 1.0)
        low, above = int(score), score - int(score)
        levels = [
            ScoreLevel(
                level=level,
                probability=1 - above if i == low else above if i == low + 1 else 0.0,
            )
            for i, level in enumerate(question.levels)
        ]
        return ScoreResult(score=score, levels=levels, confidence=max(above, 1 - above))
