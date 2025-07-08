#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wake phrase detection filter for Pipecat transcription processing.

This module provides a frame processor that filters transcription frames,
only allowing them through after wake phrases have been detected. Includes
keepalive functionality to maintain conversation flow after wake detection.
"""

import re
import time
from enum import Enum
from typing import List

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class WakeCheckFilter(FrameProcessor):
    """Frame processor that filters transcription frames based on wake phrase detection.

    This filter monitors transcription frames for configured wake phrases and only
    passes frames through after a wake phrase has been detected. Maintains a
    keepalive timeout to allow continued conversation after wake detection.
    """

    class WakeState(Enum):
        """Enumeration of wake detection states.

        Parameters:
            IDLE: No wake phrase detected, filtering active.
            AWAKE: Wake phrase detected, allowing frames through.
        """

        IDLE = 1
        AWAKE = 2

    class ParticipantState:
        """State tracking for individual participants.

        Parameters:
            participant_id: Unique identifier for the participant.
            state: Current wake state (IDLE or AWAKE).
            wake_timer: Timestamp of last wake phrase detection.
            accumulator: Accumulated text for wake phrase matching.
        """

        def __init__(self, participant_id: str):
            """Initialize participant state.

            Args:
                participant_id: Unique identifier for the participant.
            """
            self.participant_id = participant_id
            self.state = WakeCheckFilter.WakeState.IDLE
            self.wake_timer = 0.0
            self.accumulator = ""

    def __init__(self, wake_phrases: List[str], keepalive_timeout: float = 3):
        """Initialize the wake phrase filter.

        Args:
            wake_phrases: List of wake phrases to detect in transcriptions.
            keepalive_timeout: Duration in seconds to keep passing frames after
                wake detection. Defaults to 3 seconds.
        """
        super().__init__()
        self._participant_states = {}
        self._keepalive_timeout = keepalive_timeout
        self._wake_patterns = []
        for name in wake_phrases:
            pattern = re.compile(
                r"\b" + r"\s*".join(re.escape(word) for word in name.split()) + r"\b", re.IGNORECASE
            )
            self._wake_patterns.append(pattern)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, filtering transcriptions based on wake detection.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        try:
            if isinstance(frame, TranscriptionFrame):
                p = self._participant_states.get(frame.user_id)
                if p is None:
                    p = WakeCheckFilter.ParticipantState(frame.user_id)
                    self._participant_states[frame.user_id] = p

                # If we have been AWAKE within the last keepalive_timeout seconds, pass
                # the frame through
                if p.state == WakeCheckFilter.WakeState.AWAKE:
                    if time.time() - p.wake_timer < self._keepalive_timeout:
                        logger.debug(
                            f"Wake phrase keepalive timeout has not expired. Pushing {frame}"
                        )
                        p.wake_timer = time.time()
                        await self.push_frame(frame)
                        return
                    else:
                        p.state = WakeCheckFilter.WakeState.IDLE

                p.accumulator += frame.text
                for pattern in self._wake_patterns:
                    match = pattern.search(p.accumulator)
                    if match:
                        logger.debug(f"Wake phrase triggered: {match.group()}")
                        # Found the wake word. Discard from the accumulator up to the start of the match
                        # and modify the frame in place.
                        p.state = WakeCheckFilter.WakeState.AWAKE
                        p.wake_timer = time.time()
                        frame.text = p.accumulator[match.start() :]
                        p.accumulator = ""
                        await self.push_frame(frame)
                    else:
                        pass
            else:
                await self.push_frame(frame, direction)
        except Exception as e:
            error_msg = f"Error in wake word filter: {e}"
            logger.exception(error_msg)
            await self.push_error(ErrorFrame(error_msg))
