#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wake phrase detection filter for gating user interaction.

This module provides a user frame filter that blocks transcriptions, VAD events,
and interruptions until a configured wake phrase is spoken. After detection, frames
pass through freely until an inactivity timeout returns to the listening state.
"""

import asyncio
import re
from enum import Enum
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_filter.base_user_frame_filter import BaseUserFrameFilter
from pipecat.utils.asyncio.task_manager import BaseTaskManager

# Regex to strip punctuation, keeping letters, digits, underscores, and whitespace.
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


class WakePhraseUserFrameFilter(BaseUserFrameFilter):
    """User frame filter that gates interaction behind a wake phrase.

    While in the LISTENING state, transcriptions, VAD events, and interruptions
    are blocked. When a configured wake phrase is detected in a transcription,
    the filter transitions to INACTIVE (pass-through) and starts a timeout.
    If no activity occurs before the timeout expires, the filter returns to
    LISTENING.

    Event handlers available:

    - on_wake_phrase_detected: Called when a wake phrase is matched.
    - on_wake_phrase_timeout: Called when the inactivity timeout expires.

    Example::

        filter = WakePhraseUserFrameFilter(phrases=["hey robot"])

        @filter.event_handler("on_wake_phrase_detected")
        async def on_wake_phrase_detected(filter, phrase):
            ...

        @filter.event_handler("on_wake_phrase_timeout")
        async def on_wake_phrase_timeout(filter):
            ...

    """

    class State(Enum):
        """Filter states.

        Parameters:
            LISTENING: Blocking user frames, waiting for wake phrase.
            INACTIVE: Passing all frames through after wake phrase detected.
        """

        LISTENING = "listening"
        INACTIVE = "inactive"

    def __init__(
        self,
        *,
        phrases: List[str],
        timeout: float = 10.0,
        single_activation: bool = False,
        **kwargs,
    ):
        """Initialize the wake phrase filter.

        Args:
            phrases: List of wake phrases to detect.
            timeout: Inactivity timeout in seconds before returning to LISTENING.
            single_activation: If True, require wake phrase every turn. If False,
                timeout self-manages via activity detection.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)

        self._timeout = timeout
        self._single_activation = single_activation

        self._state = self.State.LISTENING

        self._timeout_event = asyncio.Event()
        self._timeout_task: Optional[asyncio.Task] = None

        # Build regex patterns from phrases.
        self._patterns = []
        for phrase in phrases:
            pattern = re.compile(
                r"\b" + r"\s*".join(re.escape(word) for word in phrase.split()) + r"\b",
                re.IGNORECASE,
            )
            self._patterns.append(pattern)

        self._register_event_handler("on_wake_phrase_detected")
        self._register_event_handler("on_wake_phrase_timeout")

    @property
    def state(self) -> "WakePhraseUserFrameFilter.State":
        """Returns the current filter state."""
        return self._state

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the filter and start the timeout task.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)

        if not self._timeout_task:
            self._timeout_task = self.task_manager.create_task(
                self._timeout_task_handler(),
                f"{self}::_timeout_task_handler",
            )

    async def cleanup(self):
        """Cleanup the filter and cancel the timeout task."""
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def reset(self):
        """Reset the filter state after a turn stop.

        With single_activation=True, transitions back to LISTENING. Otherwise
        the timeout self-manages via activity detection.
        """
        if self._single_activation:
            self._transition_to_listening()

    async def process_frame(self, frame: Frame) -> bool:
        """Process a frame and decide whether it should pass through.

        Args:
            frame: The frame to process.

        Returns:
            True if the frame should pass through, False if blocked.
        """
        if self._state == self.State.LISTENING:
            return await self._process_listening(frame)
        else:
            return await self._process_inactive(frame)

    async def _process_listening(self, frame: Frame) -> bool:
        """Process a frame while in LISTENING state.

        Blocks user interaction frames. Checks transcriptions for wake phrase.
        """
        if isinstance(frame, TranscriptionFrame):
            return await self._check_wake_phrase(frame)

        if isinstance(
            frame,
            (
                InterimTranscriptionFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                InterruptionFrame,
            ),
        ):
            return False

        # Everything else passes through.
        return True

    async def _process_inactive(self, frame: Frame) -> bool:
        """Process a frame while in INACTIVE state.

        Refreshes timeout on activity frames. All frames pass through.
        """
        if isinstance(
            frame,
            (
                UserSpeakingFrame,
                BotSpeakingFrame,
            ),
        ):
            self._refresh_timeout()

        return True

    async def _check_wake_phrase(self, frame: TranscriptionFrame) -> bool:
        """Check a transcription frame for wake phrase matches.

        Args:
            frame: The transcription frame to check.

        Returns:
            True if wake phrase matched (pass frame), False otherwise (block).
        """
        text = _PUNCTUATION_RE.sub("", frame.text)

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                matched_text = match.group()
                logger.debug(f"{self}: Wake phrase detected: '{matched_text}'")
                self._state = self.State.INACTIVE
                self._refresh_timeout()
                await self._call_event_handler("on_wake_phrase_detected", matched_text)
                return True

        return False

    def _refresh_timeout(self):
        """Signal the timeout task to restart its countdown."""
        self._timeout_event.set()

    def _transition_to_listening(self):
        """Transition to LISTENING state."""
        self._state = self.State.LISTENING

    async def _timeout_task_handler(self):
        """Background task that manages the inactivity timeout.

        Waits for activity signals and transitions to LISTENING when the
        timeout expires without activity.
        """
        while True:
            try:
                await asyncio.wait_for(
                    self._timeout_event.wait(),
                    timeout=self._timeout,
                )
                self._timeout_event.clear()
            except asyncio.TimeoutError:
                if self._state == self.State.INACTIVE:
                    logger.debug(f"{self}: Wake phrase timeout, returning to LISTENING")
                    self._transition_to_listening()
                    await self._call_event_handler("on_wake_phrase_timeout")
