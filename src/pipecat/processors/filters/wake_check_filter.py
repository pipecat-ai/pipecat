#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
import time
from enum import Enum
from typing import List

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class WakeCheckFilter(FrameProcessor):
    """This filter looks for wake phrases in the transcription frames and only passes through frames
    after a wake phrase has been detected. It also has a keepalive timeout to allow for a brief
    period of continued conversation after a wake phrase has been detected.
    """

    class WakeState(Enum):
        IDLE = 1
        AWAKE = 2

    class ParticipantState:
        def __init__(self, participant_id: str):
            self.participant_id = participant_id
            self.state = WakeCheckFilter.WakeState.IDLE
            self.wake_timer = 0.0
            self.accumulator = ""

    def __init__(self, wake_phrases: List[str], keepalive_timeout: float = 3):
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
