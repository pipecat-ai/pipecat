#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy based on VAD events only.

This strategy triggers the end of a user turn immediately when VAD indicates
the user has stopped speaking. It does not use a turn analyzer or
transcriptions, making it suitable for realtime speech-to-speech pipelines
that rely solely on VAD for turn detection.
"""

from pipecat.frames.frames import Frame, VADUserStoppedSpeakingFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


class VADUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy based on VAD (Voice Activity Detection).

    This strategy triggers the end of a user turn as soon as a VAD frame
    indicates the user has stopped speaking. It is intended for realtime
    speech-to-speech pipelines where neither a turn analyzer nor STT
    transcriptions are used to decide end of turn.
    """

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect user turn stop.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always returns CONTINUE so subsequent stop strategies are evaluated.
        """
        await super().process_frame(frame)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.trigger_user_turn_stopped()

        return ProcessFrameResult.CONTINUE
