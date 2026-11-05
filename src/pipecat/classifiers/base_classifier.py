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
results. Classifiers that can see images take :class:`ClassifierImage`
alongside the state, and classifiers that can hear take a clip of audio,
:class:`ClassifierAudio`.
"""

import io
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypeVar

from PIL import Image
from pydantic import BaseModel, Field

from pipecat.audio.utils import create_file_resampler
from pipecat.frames.frames import ImageRawFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
)
from pipecat.utils.base_object import BaseObject
from pipecat.utils.image import encode_image


class ClassifierError(Exception):
    """A classifier could not answer a question."""


class YesNoQuestion(BaseModel):
    """Whether the state meets a condition.

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

#: The image types every classifier that sees images accepts.
ClassifierImageType: TypeAlias = Literal["image/png", "image/jpeg", "image/webp"]


class ClassifierImage(BaseModel):
    """An encoded image the questions can look at, alongside the state.

    Parameters:
        data: The encoded image: PNG, JPEG or WebP.
        content_type: The image's MIME type, read from ``data`` when not given.
    """

    data: bytes = Field(repr=False)
    content_type: ClassifierImageType = Field(
        default_factory=lambda fields: ClassifierImage._content_type(fields["data"])
    )

    @classmethod
    async def from_frame(cls, frame: ImageRawFrame) -> "ClassifierImage":
        """Build an image from an image frame, such as a frame from the user's camera.

        Raw pixels are encoded as a JPEG. A frame whose format is already a
        MIME type keeps its bytes.

        Args:
            frame: The image frame.

        Returns:
            The image.
        """
        data, content_type = await encode_image(frame.image, frame.size, frame.format)
        return cls.model_validate({"data": data, "content_type": content_type})

    @staticmethod
    def _content_type(data: bytes) -> ClassifierImageType:
        """The MIME type of the encoded image, or a ValueError if it is not one classifiers take."""
        try:
            with Image.open(io.BytesIO(data)) as image:
                content_type = image.get_format_mimetype()
        except OSError as e:
            raise ValueError("ClassifierImage data is not an image") from e
        if content_type not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError(f"ClassifierImage takes PNG, JPEG or WebP, got {content_type}")
        return content_type


class ClassifierAudio(BaseModel):
    """A clip of audio the questions can hear, alongside the state.

    Parameters:
        audio: The clip, as 16-bit mono PCM.
        sample_rate: The clip's sample rate in Hz.
    """

    audio: bytes = Field(repr=False)
    sample_rate: int

    @property
    def duration_secs(self) -> float:
        """How long the clip is, in seconds."""
        return len(self.audio) / (2 * self.sample_rate)


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


class ScoreLevel(BaseModel):
    """One level of a :class:`ScoreQuestion`'s scale and how likely it is.

    Parameters:
        level: The level as the question gave it.
        probability: How likely the state is at this level, from 0 to 1.
    """

    level: str | dict[str, Any] | list[Any]
    probability: float


class ScoreResult(BaseModel):
    """Answer to a :class:`ScoreQuestion`.

    Parameters:
        score: Where the state falls on the scale, as a position from 0 (the
            first level) to one less than the number of levels. It is the
            probability-weighted position, so it may fall between two levels.
        levels: How likely each level is, in the question's order.
        confidence: How sure the classifier is of ``score``, from 0 to 1.
    """

    score: float
    levels: list[ScoreLevel]
    confidence: float

    def probability(self, level: str | dict[str, Any] | list[Any]) -> float:
        """How likely one level is.

        Args:
            level: The level as the question gave it.

        Returns:
            The level's probability.

        Raises:
            KeyError: If the scale has no such level.
        """
        for item in self.levels:
            if item.level == level:
                return item.probability
        raise KeyError(f"not a level of this scale: {level!r}")


ClassifierResult: TypeAlias = YesNoResult | ChoiceResult | ScoreResult

R = TypeVar("R", YesNoResult, ChoiceResult, ScoreResult)


class BaseClassifier(BaseObject):
    """Answers typed questions about a state.

    Every question is about a ``state``: plain text, or structured data such
    as a transcript with speaker labels or a trimmed screen snapshot.
    :meth:`ask` answers any number of questions about one state, by name;
    the three typed methods are built on it and take questions of one kind.
    Subclasses implement :meth:`_ask`.

    Classifiers that can see images say so in :attr:`supports_images`, and
    then take ``images`` with the state. The others refuse them.

    Classifiers that can hear say so in :attr:`supports_audio`, and then take
    one clip as ``audio`` with the state. They declare the clips they accept:
    from :attr:`min_audio_secs` to :attr:`max_audio_secs` long, resampled to
    :attr:`audio_sample_rate` before :meth:`_ask` gets them. A clip of another
    length is refused, and so is a clip for a classifier that cannot hear.

    An owner that runs inside a worker calls :meth:`setup` with its task
    manager before the first question, and :meth:`cleanup` when it is done.

    Event handlers available:

    - on_metrics: Called after every call with its metrics, the time it
      took and, when the classifier knows it, the tokens it used. A
      classifier cannot push frames, so the owner is the one to put them in
      a :class:`~pipecat.frames.frames.MetricsFrame`.

    Example::

        await self._classifier.setup(self.task_manager)

        @self._classifier.event_handler("on_metrics")
        async def on_metrics(classifier, data: list[MetricsData]):
            await self.push_frame(MetricsFrame(data=data))
    """

    def __init__(self, **kwargs):
        """Initialize the classifier.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._resampler = create_file_resampler()

        self._register_event_handler("on_metrics")

    @property
    def model(self) -> str | None:
        """The model that answers, named in the metrics."""
        return None

    @property
    def supports_images(self) -> bool:
        """Whether questions can come with images."""
        return False

    @property
    def supports_audio(self) -> bool:
        """Whether questions can come with a clip of audio."""
        return False

    @property
    def min_audio_secs(self) -> float:
        """The shortest clip the classifier accepts, in seconds."""
        return 0.0

    @property
    def max_audio_secs(self) -> float | None:
        """The longest clip the classifier accepts, in seconds, or None for no limit."""
        return None

    @property
    def audio_sample_rate(self) -> int | None:
        """The sample rate clips are resampled to, or None to keep their own."""
        return None

    async def ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ClassifierResult]:
        """Answer several questions about one state.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: A clip of audio the questions can hear, for a classifier
                that supports it.

        Returns:
            One result per question, by the same names, each of the type
            its question calls for.

        Raises:
            ClassifierError: If the answers could not be produced, or not in
                time: every classifier answers or raises within a bound of
                its own, so a caller waiting on it is never left hanging.
                Also if images come to a classifier that cannot see them, or
                a clip to one that cannot hear or doesn't accept its length.
        """
        if images and not self.supports_images:
            raise ClassifierError(f"{self} cannot see images")
        if audio and not self.supports_audio:
            raise ClassifierError(f"{self} cannot hear audio")
        # A clip goes to _ask only when there is one, so the _ask of a
        # classifier that cannot hear needn't take ``audio``.
        clip = {"audio": await self._clip(audio)} if audio else {}
        started = time.perf_counter()
        results, usage = await self._ask(state, questions, images=list(images or []), **clip)
        await self._call_event_handler(
            "on_metrics", self._metrics(time.perf_counter() - started, usage)
        )
        return results

    async def yes_no(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, YesNoQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, YesNoResult]:
        """Ask whether the state meets each condition.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: A clip of audio the questions can hear, for a classifier
                that supports it.

        Returns:
            How likely each answer is yes, by the same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, YesNoResult)

    async def choice(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ChoiceQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ChoiceResult]:
        """Ask which option fits the state, for each question.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: A clip of audio the questions can hear, for a classifier
                that supports it.

        Returns:
            The option that fits and how likely each one is, by the same
            names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, ChoiceResult)

    async def score(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ScoreQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ScoreResult]:
        """Ask where the state falls on each scale.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: A clip of audio the questions can hear, for a classifier
                that supports it.

        Returns:
            The position on each scale and how likely each level is, by the
            same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, ScoreResult)

    @abstractmethod
    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        images: Sequence[ClassifierImage] = (),
        audio: ClassifierAudio | None = None,
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions, and say what tokens the call used if that is known.

        ``images`` is always empty for a classifier that does not support
        images. ``audio`` is passed only for a classifier that supports it,
        and only when there is a clip: one of an accepted length, at
        :attr:`audio_sample_rate`.
        """
        pass

    async def _clip(self, audio: ClassifierAudio) -> ClassifierAudio:
        """The clip at :attr:`audio_sample_rate`, or a ClassifierError if its length isn't accepted."""
        duration = audio.duration_secs
        if duration < self.min_audio_secs:
            raise ClassifierError(
                f"{self} needs at least {self.min_audio_secs:g}s of audio, got {duration:.2f}s"
            )
        if self.max_audio_secs is not None and duration > self.max_audio_secs:
            raise ClassifierError(
                f"{self} takes at most {self.max_audio_secs:g}s of audio, got {duration:.2f}s"
            )
        if self.audio_sample_rate and audio.sample_rate != self.audio_sample_rate:
            resampled = await self._resampler.resample(
                audio.audio, audio.sample_rate, self.audio_sample_rate
            )
            audio = ClassifierAudio(audio=resampled, sample_rate=self.audio_sample_rate)
        return audio

    def _metrics(self, seconds: float, usage: LLMTokenUsage | None) -> list[MetricsData]:
        data: list[MetricsData] = [
            ProcessingMetricsData(processor=self.name, model=self.model, value=seconds)
        ]
        if usage:
            data.append(LLMUsageMetricsData(processor=self.name, model=self.model, value=usage))
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
