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
results. Questions can come with inputs alongside the state, of the kinds
the classifier takes: images (:class:`ClassifierImage`), clips of audio
(:class:`ClassifierAudio`) and files such as PDFs or videos
(:class:`ClassifierFile`).
"""

import base64
import io
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypeVar

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from pipecat.audio.utils import create_file_resampler
from pipecat.frames.frames import FileRawFrame, ImageRawFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
)
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import warn_deprecated
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


class ClassifierFile(BaseModel):
    """An encoded file the questions can look at, alongside the state, such as a PDF or a video.

    Parameters:
        content_type: The file's MIME type, such as ``application/pdf`` or
            ``video/mp4``.
        data: The file's bytes. One of ``data`` and ``url`` is required.
        url: Where the file can be fetched from, for a classifier whose
            provider fetches files itself.
    """

    content_type: str
    data: bytes | None = Field(default=None, repr=False)
    url: str | None = None

    @classmethod
    def from_frame(cls, frame: FileRawFrame) -> "ClassifierFile":
        """Build a file from a file frame, such as a file the user sent.

        Args:
            frame: The file frame, holding a base64 data URL or a URL.

        Returns:
            The file.
        """
        if frame.type == "url":
            return cls(content_type=frame.format, url=frame.file)
        return cls(content_type=frame.format, data=cls._decode(frame.file))

    @model_validator(mode="after")
    def _has_data_or_url(self) -> "ClassifierFile":
        """The file, or a ValueError if it has both data and a URL or neither."""
        if (self.data is None) == (self.url is None):
            raise ValueError("ClassifierFile takes either data or a url")
        return self

    @staticmethod
    def _decode(file: str) -> bytes:
        """The bytes of a base64 data URL, or of plain base64."""
        return base64.b64decode(file.split(",", 1)[1] if file.startswith("data:") else file)


#: What can come with the state: an image, a clip of audio or a file.
ClassifierInput: TypeAlias = ClassifierImage | ClassifierAudio | ClassifierFile


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

    Questions can come with ``inputs``: images, clips of audio and files. A
    classifier says which it takes, in :attr:`supports_images`,
    :attr:`supports_audio` and :attr:`file_types`, and refuses the rest. One
    that hears declares the clips it accepts: from :attr:`min_audio_secs` to
    :attr:`max_audio_secs` long, resampled to :attr:`audio_sample_rate`
    before :meth:`_ask` gets them. A clip of another length is refused.

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

    @property
    def file_types(self) -> frozenset[str]:
        """The MIME types of the files questions can come with; none by default."""
        return frozenset()

    async def ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        *,
        inputs: Sequence[ClassifierInput] | None = None,
        images: Sequence[ClassifierImage] | None = None,
    ) -> dict[str, ClassifierResult]:
        """Answer several questions about one state.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            inputs: Images, clips of audio and files that come with the
                state, of the kinds the classifier takes.
            images: Images the questions can look at.

                .. deprecated:: 1.16.0
                    Use ``inputs`` instead. Will be removed in 2.0.0.

        Returns:
            One result per question, by the same names, each of the type
            its question calls for.

        Raises:
            ClassifierError: If the answers could not be produced, or not in
                time: every classifier answers or raises within a bound of
                its own, so a caller waiting on it is never left hanging.
                Also if an input is of a kind the classifier doesn't take,
                or a clip of a length it doesn't accept.
        """
        checked = [await self._check(item) for item in self._inputs(inputs, images)]
        started = time.perf_counter()
        results, usage = await self._ask(state, questions, inputs=checked)
        await self._call_event_handler(
            "on_metrics", self._metrics(time.perf_counter() - started, usage)
        )
        return results

    async def yes_no(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, YesNoQuestion],
        *,
        inputs: Sequence[ClassifierInput] | None = None,
        images: Sequence[ClassifierImage] | None = None,
    ) -> dict[str, YesNoResult]:
        """Ask whether the state meets each condition.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            inputs: Images, clips of audio and files that come with the
                state, of the kinds the classifier takes.
            images: Images the questions can look at.

                .. deprecated:: 1.16.0
                    Use ``inputs`` instead. Will be removed in 2.0.0.

        Returns:
            How likely each answer is yes, by the same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, inputs=self._inputs(inputs, images))
        return self._typed(results, YesNoResult)

    async def choice(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ChoiceQuestion],
        *,
        inputs: Sequence[ClassifierInput] | None = None,
        images: Sequence[ClassifierImage] | None = None,
    ) -> dict[str, ChoiceResult]:
        """Ask which option fits the state, for each question.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            inputs: Images, clips of audio and files that come with the
                state, of the kinds the classifier takes.
            images: Images the questions can look at.

                .. deprecated:: 1.16.0
                    Use ``inputs`` instead. Will be removed in 2.0.0.

        Returns:
            The option that fits and how likely each one is, by the same
            names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, inputs=self._inputs(inputs, images))
        return self._typed(results, ChoiceResult)

    async def score(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ScoreQuestion],
        *,
        inputs: Sequence[ClassifierInput] | None = None,
        images: Sequence[ClassifierImage] | None = None,
    ) -> dict[str, ScoreResult]:
        """Ask where the state falls on each scale.

        Args:
            state: What the questions are about.
            questions: The questions, by name.
            inputs: Images, clips of audio and files that come with the
                state, of the kinds the classifier takes.
            images: Images the questions can look at.

                .. deprecated:: 1.16.0
                    Use ``inputs`` instead. Will be removed in 2.0.0.

        Returns:
            The position on each scale and how likely each level is, by the
            same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, inputs=self._inputs(inputs, images))
        return self._typed(results, ScoreResult)

    @abstractmethod
    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions, and say what tokens the call used if that is known.

        ``inputs`` holds only the kinds the classifier takes, each clip of an
        accepted length and at :attr:`audio_sample_rate`.
        """
        pass

    def _inputs(
        self,
        inputs: Sequence[ClassifierInput] | None,
        images: Sequence[ClassifierImage] | None,
    ) -> list[ClassifierInput]:
        """The inputs, with the images given the deprecated way first."""
        if images is not None:
            warn_deprecated(
                "`images` is deprecated since 1.16.0 and will be removed in 2.0.0. "
                "Use `inputs` instead.",
                stacklevel=3,
            )
        return [*(images or []), *(inputs or [])]

    async def _check(self, item: ClassifierInput) -> ClassifierInput:
        """The input as :meth:`_ask` gets it, or a ClassifierError if it isn't taken."""
        if isinstance(item, ClassifierAudio):
            return await self._clip(item)
        if isinstance(item, ClassifierImage) and not self.supports_images:
            raise ClassifierError(f"{self} cannot see images")
        if isinstance(item, ClassifierFile) and item.content_type not in self.file_types:
            raise ClassifierError(f"{self} doesn't take {item.content_type} files")
        return item

    async def _clip(self, audio: ClassifierAudio) -> ClassifierAudio:
        """The clip at :attr:`audio_sample_rate`, or a ClassifierError if it isn't accepted."""
        if not self.supports_audio:
            raise ClassifierError(f"{self} cannot hear audio")
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
