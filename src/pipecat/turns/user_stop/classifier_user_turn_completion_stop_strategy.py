#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that asks a classifier whether the turn is complete."""

import asyncio

from loguru import logger

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceQuestion, ClassifierError
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import (
    BaseUserTurnStopStrategy,
    UserTurnStoppedParams,
)

#: The question put to the classifier. The state is the user's turn so far and,
#: when the strategy has the context, the assistant's last message.
TURN_COMPLETION_QUESTION = ChoiceQuestion(
    instructions=(
        "The user is talking to a voice assistant. Their words come from speech "
        "recognition, without punctuation, and may have been cut off. Decide whether "
        "the user's turn is complete. Complete means conversationally complete, not "
        "long: one word can be a complete answer, a question is complete, a "
        "correction is complete."
    ),
    options={
        "complete": "the user has taken their turn and the assistant should answer",
        "short": (
            "the user stopped mid-sentence and will continue in a few seconds: the last "
            "words leave a phrase open, such as ending on a conjunction, a preposition, "
            "an article, or an unfinished list or number"
        ),
        "long": (
            "the user needs time to think or asked the assistant to wait, or has only "
            "acknowledged the question without answering it"
        ),
    },
)

# Events the wrapper keeps for itself: it fires them once the classifier has
# decided, so the inner detector's own must not reach the controller.
_OWN_EVENTS = ("on_user_turn_inference_triggered", "on_user_turn_stopped")


class ClassifierUserTurnCompletionStopStrategy(BaseUserTurnStopStrategy):
    """Ends the user turn only when a classifier says it is complete.

    Wraps a detector strategy, such as
    :class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`. When
    the detector decides the user stopped, this strategy asks its classifier
    whether the turn is conversationally complete instead of ending it:

    - ``complete``: the turn ends and the LLM answers.
    - ``short``: the user was cut off mid-sentence. The turn stays open for
      ``short_timeout`` seconds; if they continue and the detector fires
      again, the whole turn is classified again.
    - ``long``: the user asked for time or only acknowledged the question.
      The turn stays open for ``long_timeout`` seconds.

    When a timeout expires, or the classifier fails or takes longer than
    ``classification_timeout``, the turn ends as it would have without a
    classifier. The LLM runs once, when the turn ends, and never sees a
    turn the classifier held open.

    Give it the conversation's :class:`~pipecat.processors.aggregators.llm_context.LLMContext`
    so the classifier also sees the assistant's last message: whether "that's
    interesting" is an answer or an acknowledgement depends on what was asked.

    Example::

        stop=[
            ClassifierUserTurnCompletionStopStrategy(
                TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3()),
                classifier=JevClassifier(api_key=...),
                context=context,
            ),
        ]

    :class:`~pipecat.turns.user_turn_strategies.ClassifierUserTurnStrategies`
    wraps a whole stop chain this way.
    """

    def __init__(
        self,
        inner: BaseUserTurnStopStrategy,
        *,
        classifier: BaseClassifier,
        context: LLMContext | None = None,
        short_timeout: float = 5.0,
        long_timeout: float = 10.0,
        classification_timeout: float = 1.0,
        **kwargs,
    ):
        """Initialize the strategy.

        Args:
            inner: The detector strategy that decides when the user stopped
                speaking.
            classifier: What decides whether the turn is complete.
            context: The conversation, so the classifier sees the assistant's
                last message along with the user's turn.
            short_timeout: Seconds to keep the turn open after a ``short``
                verdict.
            long_timeout: Seconds to keep the turn open after a ``long``
                verdict.
            classification_timeout: Seconds to wait for the classifier before
                ending the turn without it. The default fits Jev; an LLM
                classifier usually needs more.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._inner = inner
        self._classifier = classifier
        self._context = context
        self._short_timeout = short_timeout
        self._long_timeout = long_timeout
        self._classification_timeout = classification_timeout

        self._text = ""
        self._params: UserTurnStoppedParams | None = None
        self._classify_task: asyncio.Task | None = None
        self._timeout_task: asyncio.Task | None = None

        self._inner.add_event_handler("on_user_turn_stopped", self._on_inner_stopped)

    @property
    def inner(self) -> BaseUserTurnStopStrategy:
        """The wrapped detector strategy."""
        return self._inner

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that decides whether a turn is complete."""
        return self._classifier

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Report what the inner strategy does with proposals."""
        return self._inner.resolves_proposed_turn_stop_frames

    def add_event_handler(self, event_name: str, handler):
        """Keep the turn events here and forward the rest to the inner strategy.

        The inner strategy's frame-side events reach listeners unchanged. Its
        inference-triggered and stopped events do not: this strategy fires its
        own once the classifier has decided.
        """
        if event_name in _OWN_EVENTS:
            super().add_event_handler(event_name, handler)
        else:
            self._inner.add_event_handler(event_name, handler)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the inner strategy and the classifier."""
        await super().setup(setup)
        await self._inner.setup(setup)
        await self._classifier.setup(setup.pipeline_worker)

    async def cleanup(self):
        """Clean up the inner strategy and the classifier."""
        await super().cleanup()
        await self._cancel_pending()
        await self._inner.cleanup()
        await self._classifier.cleanup()

    async def handle_user_turn_started(self):
        """Start a fresh turn: drop the text and any pending decision."""
        await self._cancel_pending()
        self._text = ""
        await self._inner.handle_user_turn_started()

    async def handle_user_turn_stopped(self):
        """Forget a pending decision: the turn ended some other way."""
        await self._cancel_pending()
        await self._inner.handle_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Collect the turn's text and forward the frame to the inner strategy."""
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._text = f"{self._text} {frame.text.strip()}".strip()
        return await self._inner.process_frame(frame)

    async def _on_inner_stopped(self, strategy, params: UserTurnStoppedParams):
        """The detector says the user stopped: ask the classifier."""
        self._params = params
        await self._cancel_pending()
        self._classify_task = self.task_manager.create_task(self._classify(), f"{self}::_classify")

    async def _classify(self):
        state: dict[str, str] = {"user": self._text}
        assistant = self._last_assistant_message()
        if assistant:
            state = {"assistant": assistant, "user": self._text}
        try:
            results = await asyncio.wait_for(
                self._classifier.choice(state, {"turn": TURN_COMPLETION_QUESTION}),
                timeout=self._classification_timeout,
            )
            result = results["turn"]
        except (ClassifierError, TimeoutError) as e:
            logger.warning(f"{self}: no classification, ending the turn: {e}")
            await self._end_turn()
            return

        logger.debug(f"{self}: {result.choice} ({result.confidence:.2f}) for [{self._text}]")
        if result.choice == "short":
            self._timeout_task = self.task_manager.create_task(
                self._timeout(self._short_timeout), f"{self}::_short_timeout"
            )
        elif result.choice == "long":
            self._timeout_task = self.task_manager.create_task(
                self._timeout(self._long_timeout), f"{self}::_long_timeout"
            )
        else:
            await self._end_turn()

    async def _timeout(self, seconds: float):
        await asyncio.sleep(seconds)
        logger.debug(f"{self}: the user did not continue after {seconds}s, ending the turn")
        await self._end_turn()

    async def _end_turn(self):
        params = self._params
        await self.trigger_user_turn_stopped(
            enable_user_speaking_frames=params.enable_user_speaking_frames if params else None
        )

    async def _cancel_pending(self):
        if self._classify_task and self._classify_task is not asyncio.current_task():
            await self.cancel_task(self._classify_task)
        self._classify_task = None
        if self._timeout_task and self._timeout_task is not asyncio.current_task():
            await self.cancel_task(self._timeout_task)
        self._timeout_task = None

    def _last_assistant_message(self) -> str | None:
        if self._context is None:
            return None
        for message in reversed(self._context.messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(t for t in texts if t) or None
            return None
        return None
