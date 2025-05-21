#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class UserBotLatencyLogObserver(BaseObserver):
    """Observer that logs the latency between when the user stops speaking and
    when the bot starts speaking.

    This helps measure how quickly the AI services respond.

    """

    def __init__(self):
        super().__init__()
        self._processed_frames = set()
        self._user_stopped_time = 0

    async def on_push_frame(self, data: FramePushed):
        # Only process downstream frames
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, UserStartedSpeakingFrame):
            self._user_stopped_time = 0
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            self._user_stopped_time = time.time()
        elif isinstance(data.frame, BotStartedSpeakingFrame) and self._user_stopped_time:
            latency = time.time() - self._user_stopped_time
            logger.debug(f"⏱️ LATENCY FROM USER STOPPED SPEAKING TO BOT STARTED SPEAKING: {latency}")
