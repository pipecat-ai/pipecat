#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by the Zoom Decisions API.

:class:`ZoomDecisionsClassifier` asks every question in one request, through a
:class:`~pipecat.classifiers.zoom.decisions.client.ZoomDecisionsClient`.
Questions and answers are converted to and from Zoom's format by
:mod:`pipecat.services.zoom_ai.decisions`.
"""

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ClassifierError,
    ClassifierImage,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    ScoreQuestion,
)
from pipecat.classifiers.zoom.decisions.client import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    ZoomDecisionsClient,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.zoom_ai.decisions import questions_to_wire, result_from_wire
from pipecat.utils.asyncio.task_manager import BaseTaskManager

#: The most questions the Decisions API takes in one request.
ZOOM_DECISIONS_MAX_QUESTIONS = 10

#: The most options the Decisions API takes in one choice question.
ZOOM_DECISIONS_MAX_CHOICE_OPTIONS = 255

#: The most levels the Decisions API takes in one score question.
ZOOM_DECISIONS_MAX_SCORE_LEVELS = 10

#: The most images the Decisions API takes in one request.
ZOOM_DECISIONS_MAX_IMAGES = 4


class ZoomDecisionsClassifier(BaseClassifier):
    """Answers questions by asking the Zoom Decisions API.

    Structured state, instructions and descriptions are sent as JSON. Build
    one with an API key to get a client of its own, or pass a
    :class:`ZoomDecisionsClient` to share one between several classifiers. A
    client the classifier created is closed in :meth:`cleanup`; a shared one is
    left to whoever made it.

    One request takes up to :data:`ZOOM_DECISIONS_MAX_QUESTIONS` questions, a
    choice question up to :data:`ZOOM_DECISIONS_MAX_CHOICE_OPTIONS` options,
    and a score question up to :data:`ZOOM_DECISIONS_MAX_SCORE_LEVELS` levels.
    The Decisions API can see images: up to :data:`ZOOM_DECISIONS_MAX_IMAGES`
    per call.

    Example::

        classifier = ZoomDecisionsClassifier(api_key=os.getenv("ZOOM_AI_API_KEY"))
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: ZoomDecisionsClient | None = None,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: Zoom AI Services API key, when the classifier should have
                a client of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
            model: The decision model a client of its own asks:
                ``zoom-decisions`` or ``zoom-decisions-pro``.
            url: Where a client of its own sends its questions.
            timeout: Seconds a client of its own waits for an answer, retries
                included.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key:
                raise ValueError(
                    "ZoomDecisionsClassifier needs an API key or a ZoomDecisionsClient"
                )
            client = ZoomDecisionsClient(api_key=api_key, model=model, url=url, timeout=timeout)
        self._client = client

    @property
    def client(self) -> ZoomDecisionsClient:
        """The client this classifier asks through."""
        return self._client

    async def setup(self, task_manager: BaseTaskManager):
        """Open the connection to Zoom ahead of the first question.

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

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        images = [item for item in inputs if isinstance(item, ClassifierImage)]
        self._check_limits(questions, images)
        # Zoom reads images only when the request asks it to.
        options = {"enable_image_input": True} if images else {}
        answers, usage = await self._client.ask(
            self._input(state, images), questions_to_wire(questions), **options
        )
        results = {
            name: result_from_wire(name, question, answers[name])
            for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            cache_read_input_tokens=usage.cached_input_tokens or None,
        )

    def _check_limits(
        self, questions: Mapping[str, ClassifierQuestion], images: Sequence[ClassifierImage]
    ):
        """Raise a ClassifierError if the request is more than Zoom takes in one."""
        if len(images) > ZOOM_DECISIONS_MAX_IMAGES:
            raise ClassifierError(
                f"Zoom Decisions takes at most {ZOOM_DECISIONS_MAX_IMAGES} images, "
                f"got {len(images)}"
            )
        if len(questions) > ZOOM_DECISIONS_MAX_QUESTIONS:
            raise ClassifierError(
                f"Zoom Decisions takes at most {ZOOM_DECISIONS_MAX_QUESTIONS} questions "
                f"per request, got {len(questions)}"
            )
        for name, question in questions.items():
            if (
                isinstance(question, ChoiceQuestion)
                and len(question.options) > ZOOM_DECISIONS_MAX_CHOICE_OPTIONS
            ):
                raise ClassifierError(
                    f"Zoom Decisions takes at most {ZOOM_DECISIONS_MAX_CHOICE_OPTIONS} "
                    f"options, got {len(question.options)} for {name!r}"
                )
            if (
                isinstance(question, ScoreQuestion)
                and len(question.levels) > ZOOM_DECISIONS_MAX_SCORE_LEVELS
            ):
                raise ClassifierError(
                    f"Zoom Decisions takes at most {ZOOM_DECISIONS_MAX_SCORE_LEVELS} "
                    f"levels, got {len(question.levels)} for {name!r}"
                )

    def _input(
        self, state: str | dict[str, Any] | list[Any], images: Sequence[ClassifierImage]
    ) -> str | list[dict[str, Any]]:
        """The request's input: the state as text, or a user message with any images first."""
        text = state if isinstance(state, str) else json.dumps(state, indent=2)
        if not images:
            return text
        content: list[dict[str, Any]] = [
            {
                "type": "input_image",
                "image_url": f"data:{image.content_type};base64,"
                f"{base64.b64encode(image.data).decode()}",
            }
            for image in images
        ]
        content.append({"type": "input_text", "text": text})
        return [{"role": "user", "content": content}]
