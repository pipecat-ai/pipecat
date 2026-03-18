#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on transcriptions."""

from typing import Optional

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.process_frame_result import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class TranscriptionUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on transcriptions.

    This strategy signals the start of a user turn when a transcription is
    received while the bot is speaking. It is useful as a fallback in scenarios
    where VAD-based detection fails (for example, when the user speaks very
    softly) but the STT service still produces transcriptions.

    """

    def __init__(self, *, use_interim: bool = True, **kwargs):
        """Initialize transcription-based user turn start strategy."""
        super().__init__(**kwargs)
        self._use_interim = use_interim

    async def process_frame(self, frame: Frame) -> Optional[ProcessFrameResult]:
        """Process an incoming frame to detect the start of a user turn.

        Args:
            frame: The frame to be processed.

        Returns:
            STOP if a transcription was received, CONTINUE otherwise.
        """
        if isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        elif isinstance(frame, TranscriptionFrame):
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        return ProcessFrameResult.CONTINUE
