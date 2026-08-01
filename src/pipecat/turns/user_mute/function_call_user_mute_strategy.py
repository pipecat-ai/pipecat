#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that mutes the user while a function call is executing."""

from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.utils.deprecation import deprecated


@deprecated(
    "`FunctionCallUserMuteStrategy` is deprecated since 1.7.0 and will be removed in 2.0.0. "
    "Use `@tool_options(cancel_on_interruption=False)` instead."
)
class FunctionCallUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that mutes the user while a function call is executing.

    This strategy ensures that user input does not interfere with ongoing
    function execution. While a function call is active, all user frames are
    muted. Once the function call completes or is canceled, user input is
    allowed again.

    .. deprecated:: 1.7.0
        Use :func:`~pipecat.adapters.schemas.direct_function.tool_options` with
        ``cancel_on_interruption=False`` instead, which keeps a tool call running
        across an interruption without suppressing user speech. Muting covers
        every call, so the user is silenced for as long as the slowest one takes.
        Will be removed in 2.0.0.
    """

    def __init__(self):
        """Initialize the function call user mute strategy."""
        super().__init__()
        self._function_call_in_progress: set[str] = set()

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
            # Untracked ids reach here: cancel_async_tool_call is excluded from
            # FunctionCallsStartedFrame yet still emits a result, async tools
            # emit a result frame per intermediate update, and a bus bridge can
            # re-deliver a result another worker already handled.
            self._function_call_in_progress.discard(frame.tool_call_id)

        return bool(self._function_call_in_progress)

    async def _handle_function_calls_started(self, frame: FunctionCallsStartedFrame):
        for f in frame.function_calls:
            self._function_call_in_progress.add(f.tool_call_id)
