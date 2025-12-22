#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Time-based bot turn start strategy."""

import asyncio
from typing import Optional

from pipecat.frames.frames import Frame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.bot.base_bot_turn_start_strategy import BaseBotTurnStartStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TimeoutBotTurnStartStrategy(BaseBotTurnStartStrategy):
    """Bot turn start strategy based on a timeout.

    This strategy starts the bot turn after a fixed timeout once the user is no
    longer speaking. It is intended as a fallback strategy when other turn start
    signals (e.g. transcription-based strategies) are unavailable or unreliable.

    For example, if VAD detects that the user stopped speaking but no further
    events arrive, this strategy ensures the bot can continue.

    """

    def __init__(self, *, timeout: float = 5.0):
        """Initialize the bot turn start strategy.

        Args:
            timeout: Time in seconds to wait before considering the user's turn
                finished and starting the bot turn.
        """
        super().__init__()
        self._timeout = timeout
        self._vad_user_speaking = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._vad_user_speaking = False
        self._event.clear()

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)
        self._task = task_manager.create_task(self._task_handler(), f"{self}::_task_handler")

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to update strategy state.

        Args:
            frame: The frame to be analyzed.

        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._event.set()

    async def _task_handler(self):
        """Background task that triggers the bot turn after a timeout."""
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except asyncio.TimeoutError:
                if not self._vad_user_speaking:
                    await self.trigger_bot_turn_started()
