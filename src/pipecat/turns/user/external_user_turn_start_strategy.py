#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy triggered by externally emitted frames."""

from pipecat.frames.frames import Frame, UserStartedSpeakingFrame
from pipecat.turns.user.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class ExternalUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy controlled by an external processor.

    This strategy does not determine when a user turn starts on its own, instead
    it relies on a different processor in the pipeline which is responsible for
    emitting `UserStartedSpeakingFrame` frames.

    """

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect user turn start.

        Args:
            frame: The frame to be analyzed.
        """
        await super().process_frame(frame)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.trigger_user_turn_started(enable_user_speaking_frames=False)
