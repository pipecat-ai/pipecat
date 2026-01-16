#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that mutes the user while a function call is executing."""

from typing import Set

from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class FunctionCallUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that mutes the user while a function call is executing.

    This strategy ensures that user input does not interfere with ongoing
    function execution. While a function call is active, all user frames are
    muted. Once the function call completes or is canceled, user input is
    allowed again.

    """

    def __init__(self):
        """Initialize the function call user mute strategy."""
        super().__init__()
        self._function_call_in_progress: Set[str] = set()

    async def reset(self):
        """Reset the strategy to its initial state."""
        self._function_call_in_progress = set()

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        await super().process_frame(frame)

        if isinstance(frame, FunctionCallsStartedFrame):
            await self._handle_function_calls_started(frame)
        elif isinstance(frame, (FunctionCallCancelFrame, FunctionCallResultFrame)):
            self._function_call_in_progress.remove(frame.tool_call_id)

        return bool(self._function_call_in_progress)

    async def _handle_function_calls_started(self, frame: FunctionCallsStartedFrame):
        for f in frame.function_calls:
            self._function_call_in_progress.add(f.tool_call_id)
