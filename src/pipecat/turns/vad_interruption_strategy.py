#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interruption strategy based on VAD events."""

from pipecat.frames.frames import Frame, VADUserStartedSpeakingFrame
from pipecat.turns.base_interruption_strategy import BaseInterruptionStrategy


class VADInterruptionStrategy(BaseInterruptionStrategy):
    """Interruption strategy based on VAD.

    This is an interruption strategy based simply on VAD. As soon as the VAD
    detects the user is speaking we will emit an interruption.
    """

    def __init__(self):
        """Initialize the base interruption strategy."""
        super().__init__()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame.

        The analysis of incoming frames will decide if the bot should be interrupted.

        Args:
            frame: The frame to be processed.
        """
        await super().process_frame(frame)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._call_event_handler("on_should_interrupt")
