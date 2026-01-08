#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on VAD events."""

from pipecat.frames.frames import Frame, VADUserStartedSpeakingFrame
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class VADUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on VAD (Voice Activity Detection).

    This strategy assumes the user turn starts as soon as a VAD frame indicates
    that the user has started speaking.

    """

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect user turn start.

        Args:
            frame: The frame to be analyzed.
        """
        await super().process_frame(frame)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.trigger_user_turn_started()
