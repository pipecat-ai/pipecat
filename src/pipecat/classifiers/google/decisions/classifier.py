#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Google's Decisions API.

:class:`GoogleDecisionsClassifier` turns each question into a request and
each reply into a result, through a
:class:`~pipecat.classifiers.google.decisions.client.GoogleDecisionsClient`.
"""

import base64
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
    ClassifierImage,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    ScoreLevel,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.google.decisions.client import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    GoogleDecisionsClient,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The most options the Decisions API takes in one choice question.
GOOGLE_DECISIONS_MAX_CHOICE_OPTIONS = 255

#: Every file type the Decisions API reads, by MIME type, and the content block it goes in.
GOOGLE_DECISIONS_FILE_TYPES: dict[str, str] = {
    "application/pdf": "document",
    **{
        f"image/{kind}": "image"
        for kind in ("png", "jpeg", "webp", "heic", "heif", "gif", "bmp", "tiff")
    },
    **{
        f"audio/{kind}": "audio"
        for kind in (
            "wav",
            "mp3",
            "mpeg",
            "aiff",
            "aac",
            "ogg",
            "flac",
            "m4a",
            "opus",
            "webm",
            "l16",
            "alaw",
            "mulaw",
        )
    },
    **{
        f"video/{kind}": "video"
        for kind in ("mp4", "mpeg", "mov", "avi", "webm", "wmv", "x-flv", "3gpp")
    },
}


class GoogleDecisionsClassifier(BaseClassifier):
    """Answers questions by asking Google's Decisions API.

    Structured state, instructions and descriptions are sent as JSON text.
    Build one with a Gemini API key to get a client of its own, or pass a
    :class:`GoogleDecisionsClient` to share one between several classifiers.
    A client the classifier created is closed in :meth:`cleanup`; a shared
    one is left to whoever made it.

    The Decisions API sees images, hears clips of audio, sent as 16 kHz WAV,
    and reads files of the types in :data:`GOOGLE_DECISIONS_FILE_TYPES`:
    PDFs, videos, and images and audio in other formats, as bytes or by URL.
    The inputs follow the state in the order they're given.

    Example::

        classifier = GoogleDecisionsClassifier(api_key=os.getenv("GOOGLE_API_KEY"))
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: GoogleDecisionsClient | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: Gemini API key, when the classifier should have a client
                of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
            model: The decision model a client of its own asks.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key:
                raise ValueError(
                    "GoogleDecisionsClassifier needs an API key or a GoogleDecisionsClient"
                )
            client = GoogleDecisionsClient(api_key=api_key, model=model, timeout=timeout)
        self._client = client

    @property
    def client(self) -> GoogleDecisionsClient:
        """The client this classifier asks through."""
        return self._client

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to Google ahead of the first question.

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

    @property
    def supports_images(self) -> bool:
        """The Decisions API can see images."""
        return True

    @property
    def supports_audio(self) -> bool:
        """The Decisions API can hear audio."""
        return True

    @property
    def audio_sample_rate(self) -> int:
        """Clips are sent at 16 kHz, which keeps requests small."""
        return 16000

    @property
    def file_types(self) -> frozenset[str]:
        """The Decisions API reads PDFs, videos, and images and audio in many formats."""
        return frozenset(GOOGLE_DECISIONS_FILE_TYPES)

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        answers, usage = await self._client.ask(
            self._input(state, inputs),
            {name: self._to_google(question) for name, question in questions.items()},
        )
        results = {
            name: self._from_google(question, answers[name]) for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    def _input(
        self, state: str | dict[str, Any] | list[Any], inputs: Sequence[ClassifierInput]
    ) -> str | list[dict[str, Any]]:
        """The request's input: the state as text, then a block for each input, in order."""
        text = self._text(state)
        if not inputs:
            return text
        return [{"type": "text", "text": text}, *(self._block(item) for item in inputs)]

    def _block(self, item: ClassifierInput) -> dict[str, Any]:
        """An input as one of the Decisions API's content blocks."""
        if isinstance(item, ClassifierImage):
            return {
                "type": "image",
                "data": self._base64(item.data),
                "mime_type": item.content_type,
            }
        if isinstance(item, ClassifierAudio):
            wav = pcm_to_wav(item.audio, item.sample_rate)
            return {"type": "audio", "data": self._base64(wav), "mime_type": "audio/wav"}
        source = {"data": self._base64(item.data)} if item.data is not None else {"uri": item.url}
        return {
            "type": GOOGLE_DECISIONS_FILE_TYPES[item.content_type],
            "mime_type": item.content_type,
            **source,
        }

    def _base64(self, data: bytes) -> str:
        """Bytes as the base64 text the Decisions API takes."""
        return base64.b64encode(data).decode()

    def _to_google(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in the Decisions API's own format."""
        google: dict[str, Any] = {"instructions": self._text(question.instructions)}
        if isinstance(question, YesNoQuestion):
            google["type"] = "noul"
            criteria = {
                key: self._text(value)
                for key, value in (
                    ("true_description", question.yes),
                    ("false_description", question.no),
                )
                if value is not None
            }
            if criteria:
                google["criteria"] = criteria
        elif isinstance(question, ChoiceQuestion):
            if len(question.options) > GOOGLE_DECISIONS_MAX_CHOICE_OPTIONS:
                raise ClassifierError(
                    f"Google Decisions takes at most {GOOGLE_DECISIONS_MAX_CHOICE_OPTIONS} "
                    f"options, got {len(question.options)}"
                )
            google["type"] = "choice"
            google["criteria"] = {
                option: "" if description is None else self._text(description)
                for option, description in question.options.items()
            }
        else:
            google["type"] = "score"
            google["criteria"] = [self._text(level) for level in question.levels]
        return google

    def _from_google(self, question: ClassifierQuestion, answer: Any) -> ClassifierResult:
        """A result built from the Decisions API's answer to the question."""
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._number(answer, "noul"))
        if isinstance(question, ChoiceQuestion):
            choice = str(getattr(answer, "choice", None))
            if choice not in question.options:
                raise ClassifierError(f"Google Decisions chose {choice!r}, which is not an option")
            return ChoiceResult(
                choice=choice,
                probabilities=self._probabilities(answer, list(question.options)),
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
        return value if isinstance(value, str) else json.dumps(value)

    def _number(self, answer: Any, field: str) -> float:
        """The number the API gave as ``field``, or a ClassifierError if it gave none."""
        try:
            return float(getattr(answer, field))
        except (AttributeError, TypeError, ValueError) as e:
            raise ClassifierError(f"Google Decisions reply has no usable '{field}'") from e

    def _probabilities(self, answer: Any, keys: list[str]) -> dict[str, float]:
        """The probability the API gave for each key, 0 for the ones it left out."""
        given = getattr(answer, "probabilities", None)
        given = given if isinstance(given, dict) else {}
        try:
            return {key: float(given.get(key, 0.0)) for key in keys}
        except (TypeError, ValueError) as e:
            raise ClassifierError(f"Google Decisions reply has an unusable probability: {e}") from e
