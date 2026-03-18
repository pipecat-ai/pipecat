#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy that gates interaction behind wake phrase detection."""

import asyncio
import enum
import re
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    TranscriptionFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.process_frame_result import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class _WakeState(enum.Enum):
    """Internal state for wake phrase detection."""

    LISTENING = "listening"
    INACTIVE = "inactive"


class WakePhraseUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy that requires a wake phrase before interaction.

    Blocks subsequent strategies until a wake phrase is detected in a final
    transcription. After detection, allows interaction for a configurable
    timeout period before requiring the wake phrase again. Use
    ``single_activation=True`` to require the wake phrase before every turn.

    This strategy should be placed first in the start strategies list.

    Event handlers available:

    - on_wake_phrase_detected: Called when a wake phrase is matched.
    - on_wake_phrase_timeout: Called when the inactivity timeout expires
      (timeout mode only).

    Example::

        # Timeout mode (default): wake phrase unlocks interaction for 10s
        strategy = WakePhraseUserTurnStartStrategy(
            phrases=["hey pipecat", "ok pipecat"],
            timeout=10.0,
        )

        # Single activation: wake phrase required before every turn
        strategy = WakePhraseUserTurnStartStrategy(
            phrases=["hey pipecat"],
            single_activation=True,
        )

        @strategy.event_handler("on_wake_phrase_detected")
        async def on_wake_phrase_detected(strategy, phrase):
            ...

        @strategy.event_handler("on_wake_phrase_timeout")
        async def on_wake_phrase_timeout(strategy):
            ...

    Args:
        phrases: List of wake phrases to detect.
        timeout: Inactivity timeout in seconds before returning to LISTENING.
            Ignored when ``single_activation=True``.
        single_activation: If True, the wake phrase is required before every
            turn. The strategy returns to LISTENING after each turn completes.
        **kwargs: Additional keyword arguments passed to parent.
    """

    def __init__(
        self,
        *,
        phrases: List[str],
        timeout: float = 10.0,
        single_activation: bool = False,
        **kwargs,
    ):
        """Initialize the wake phrase user turn start strategy.

        Args:
            phrases: List of wake phrases to detect.
            timeout: Inactivity timeout in seconds before returning to LISTENING.
                Ignored when ``single_activation=True``.
            single_activation: If True, the wake phrase is required before every
                turn. The strategy returns to LISTENING after each turn completes.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._phrases = phrases
        self._timeout = timeout
        self._single_activation = single_activation

        self._patterns: List[re.Pattern] = []
        for phrase in phrases:
            pattern = re.compile(
                r"\b" + r"\s*".join(re.escape(word) for word in phrase.split()) + r"\b",
                re.IGNORECASE,
            )
            self._patterns.append(pattern)

        self._state = _WakeState.LISTENING
        self._accumulated_text = ""

        self._timeout_event = asyncio.Event()
        self._timeout_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_wake_phrase_detected")
        self._register_event_handler("on_wake_phrase_timeout")

    @property
    def state(self) -> _WakeState:
        """Returns the current wake state."""
        return self._state

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)
        if not self._single_activation and not self._timeout_task:
            self._timeout_task = self.task_manager.create_task(
                self._timeout_task_handler(),
                f"{self}::_timeout_task_handler",
            )

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def reset(self):
        """Reset the strategy.

        In single activation mode, transitions back to LISTENING so the wake
        phrase is required again for the next turn. In timeout mode, preserves
        state and refreshes timeout since reset means a turn started (activity).
        """
        await super().reset()
        if self._state == _WakeState.INACTIVE:
            if self._single_activation:
                self._state = _WakeState.LISTENING
                self._accumulated_text = ""
            else:
                self._refresh_timeout()

    async def process_frame(self, frame: Frame) -> Optional[ProcessFrameResult]:
        """Process an incoming frame for wake phrase detection or passthrough.

        Args:
            frame: The frame to be processed.

        Returns:
            STOP when the wake phrase is detected or when in LISTENING state
            (blocks subsequent strategies), CONTINUE when in INACTIVE state
            (allows subsequent strategies to proceed).
        """
        await super().process_frame(frame)

        if self._state == _WakeState.LISTENING:
            return await self._process_listening(frame)
        else:
            return await self._process_inactive(frame)

    async def _process_listening(self, frame: Frame) -> ProcessFrameResult:
        """Process a frame while in LISTENING state.

        Only final ``TranscriptionFrame`` instances are checked for wake phrase
        matches. When a match is found, a user turn start is triggered.
        Transcription frames that don't match have their text cleared so that
        pre-wake-phrase speech is not added to the LLM context. All frames
        return STOP to block subsequent strategies.
        """
        if isinstance(frame, TranscriptionFrame):
            if self._check_wake_phrase(frame.text):
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            await self.trigger_reset_aggregation()

        return ProcessFrameResult.STOP

    async def _process_inactive(self, frame: Frame) -> ProcessFrameResult:
        """Process a frame while in INACTIVE state.

        Refreshes the timeout on activity frames (timeout mode only). Returns
        CONTINUE so subsequent strategies can process the frame.
        """
        if not self._single_activation:
            if isinstance(frame, (UserSpeakingFrame, BotSpeakingFrame)):
                self._refresh_timeout()
            elif isinstance(frame, TranscriptionFrame):
                self._refresh_timeout()
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                self._refresh_timeout()

        return ProcessFrameResult.CONTINUE

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        """Strip punctuation from text, keeping only letters, digits, and whitespace."""
        return re.sub(r"[^\w\s]", "", text)

    def _check_wake_phrase(self, text: str) -> bool:
        """Check if the accumulated text contains a wake phrase.

        Punctuation is stripped before matching so that STT output like
        "Hey, Pipecat!" still matches the phrase "hey pipecat".

        Args:
            text: New transcription text to append and check.

        Returns:
            True if a wake phrase was found, False otherwise.
        """
        self._accumulated_text += " " + self._strip_punctuation(text)
        # Cap accumulated text to prevent unbounded growth.
        if len(self._accumulated_text) > 250:
            self._accumulated_text = self._accumulated_text[-250:]

        for i, pattern in enumerate(self._patterns):
            if pattern.search(self._accumulated_text):
                phrase = self._phrases[i]
                logger.debug(f"{self} wake phrase detected: {phrase!r}")
                self._transition_to_inactive(phrase)
                return True

        return False

    def _transition_to_inactive(self, phrase: str):
        """Transition from LISTENING to INACTIVE state."""
        self._state = _WakeState.INACTIVE
        self._accumulated_text = ""
        if not self._single_activation:
            self._refresh_timeout()
        self.task_manager.create_task(
            self._call_event_handler("on_wake_phrase_detected", phrase),
            f"{self}::on_wake_phrase_detected",
        )

    def _transition_to_listening(self):
        """Transition from INACTIVE to LISTENING state."""
        logger.debug(f"{self} wake phrase timeout, returning to LISTENING")
        self._state = _WakeState.LISTENING
        self._accumulated_text = ""
        self.task_manager.create_task(
            self._call_event_handler("on_wake_phrase_timeout"),
            f"{self}::on_wake_phrase_timeout",
        )

    def _refresh_timeout(self):
        """Refresh the inactivity timeout."""
        self._timeout_event.set()

    async def _timeout_task_handler(self):
        """Background task that monitors inactivity timeout."""
        while True:
            try:
                await asyncio.wait_for(
                    self._timeout_event.wait(),
                    timeout=self._timeout,
                )
                self._timeout_event.clear()
            except asyncio.TimeoutError:
                if self._state == _WakeState.INACTIVE:
                    self._transition_to_listening()
