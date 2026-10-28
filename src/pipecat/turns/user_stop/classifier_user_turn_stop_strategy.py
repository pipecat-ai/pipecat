#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that asks a classifier whether the turn is complete."""

import asyncio

from loguru import logger

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceQuestion, ClassifierError
from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

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

_SPEECH_STARTED = (
    VADUserStartedSpeakingFrame,
    UserStartedSpeakingFrame,
    ProposedUserStartedSpeakingFrame,
)
_SPEECH_STOPPED = (
    VADUserStoppedSpeakingFrame,
    UserStoppedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
)


class ClassifierUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Decides the user is done when a classifier says their turn is complete.

    Each time the user pauses with new words transcribed, the strategy asks
    its classifier whether the turn so far is conversationally complete:

    - ``complete``: it reports that the user is done.
    - ``short``: the user was cut off mid-sentence. It reports after
      ``short_timeout`` seconds unless they continue.
    - ``long``: the user asked for time or only acknowledged the question. It
      reports after ``long_timeout`` seconds unless they continue.

    If the classifier fails or takes longer than ``classification_timeout``, it
    reports right away, so a turn never waits on a classifier that can't
    answer. A pause with no words transcribed isn't classified, and the turn
    controller's stop timeout ends such a turn.

    Next to a turn analyzer such as
    :class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`, in a
    ``StrategyMode.ALL`` list, the default, the turn ends once both have
    decided: the analyzer hears that more is coming after words that read
    complete, but holds a complete turn until it decides too. As the only stop
    strategy, it ends a turn as soon as the classifier finds it complete. In a
    ``StrategyMode.FIRST`` list the classifier can only end a turn sooner.

    Give it the conversation's :class:`~pipecat.processors.aggregators.llm_context.LLMContext`
    so the classifier also sees the assistant's last message: whether "that's
    interesting" is an answer or an acknowledgement depends on what was asked.

    Example::

        stop=[
            ClassifierUserTurnStopStrategy(
                classifier=JevClassifier(api_key=...),
                context=context,
            ),
        ]

    :class:`~pipecat.turns.user_turn_strategies.ClassifierUserTurnStrategies`
    adds it after the default stop strategies.
    """

    def __init__(
        self,
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
            classifier: What decides whether the turn is complete.
            context: The conversation, so the classifier sees the assistant's
                last message along with the user's turn.
            short_timeout: Seconds to wait after a ``short`` verdict before
                reporting that the user is done.
            long_timeout: Seconds to wait after a ``long`` verdict before
                reporting that the user is done.
            classification_timeout: Seconds to wait for the classifier before
                reporting without it.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._classifier = classifier
        self._context = context
        self._short_timeout = short_timeout
        self._long_timeout = long_timeout
        self._classification_timeout = classification_timeout

        self._turn_open = False
        self._user_speaking = False
        self._text = ""
        # The turn text last given to the classifier, so a pause without new
        # words doesn't ask again.
        self._classified_text: str | None = None
        self._classify_task: asyncio.Task | None = None
        self._wait_task: asyncio.Task | None = None

        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that decides whether a turn is complete."""
        return self._classifier

    @property
    def decision_scheduled(self) -> bool:
        """Whether a ``short`` or ``long`` wait is running, which decides when it ends."""
        return self._wait_task is not None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the strategy and its classifier."""
        await super().setup(setup)
        await self._classifier.setup(setup.task_manager)

    async def cleanup(self):
        """Clean up the strategy and its classifier."""
        await super().cleanup()
        await self._cancel_pending()
        await self._classifier.cleanup()

    async def handle_user_turn_started(self):
        """Start a fresh turn with no text."""
        await self._cancel_pending()
        self._turn_open = True
        self._text = ""
        self._classified_text = None

    async def handle_user_turn_stopped(self):
        """Forget the turn and any pending decision."""
        await self._cancel_pending()
        self._turn_open = False
        self._text = ""
        self._classified_text = None

    async def process_frame(self, frame: Frame):
        """Collect the turn's text and classify it when the user pauses."""
        if isinstance(frame, _SPEECH_STARTED):
            await self._handle_speech_started()
        elif isinstance(frame, _SPEECH_STOPPED):
            self._user_speaking = False
            await self._maybe_classify()
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._text = f"{self._text} {frame.text.strip()}".strip()
            await self._maybe_classify()

    async def _handle_speech_started(self):
        # The controller clears the stop decisions on speech, so the turn is
        # classified again at the next pause, even without new words. A verdict
        # still on its way is about words the user has gone past.
        self._user_speaking = True
        self._classified_text = None
        await self._cancel_pending()

    async def _maybe_classify(self):
        if not self._turn_open or self._user_speaking or not self._text:
            return
        if self._text == self._classified_text:
            return

        await self._cancel_pending()
        self._classified_text = self._text
        self._classify_task = self.create_task(self._classify(self._text), f"{self}::_classify")

    async def _classify(self, text: str):
        state: dict[str, str] = {"user": text}
        assistant = self._last_assistant_message()
        if assistant:
            state = {"assistant": assistant, "user": text}
        try:
            results = await asyncio.wait_for(
                self._classifier.choice(state, {"turn": TURN_COMPLETION_QUESTION}),
                timeout=self._classification_timeout,
            )
            result = results["turn"]
        except (ClassifierError, TimeoutError) as e:
            logger.warning(f"{self}: no classification, leaving the turn to the others: {e}")
            self._classify_task = None
            await self.report_user_turn_stopped()
            return

        logger.debug(f"{self}: {result.choice} ({result.confidence:.2f}) for [{text}]")
        self._classify_task = None
        if result.choice == "short":
            self._wait_task = self.create_task(self._wait(self._short_timeout), f"{self}::_wait")
        elif result.choice == "long":
            self._wait_task = self.create_task(self._wait(self._long_timeout), f"{self}::_wait")
        else:
            await self.report_user_turn_stopped()

    async def _wait(self, seconds: float):
        await asyncio.sleep(seconds)
        logger.debug(f"{self}: the user did not continue after {seconds}s")
        self._wait_task = None
        await self.report_user_turn_stopped()

    async def _cancel_pending(self):
        if self._classify_task:
            await self.cancel_task(self._classify_task)
            self._classify_task = None
        if self._wait_task:
            await self.cancel_task(self._wait_task)
            self._wait_task = None

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        await self.push_frame(MetricsFrame(data=data))

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
