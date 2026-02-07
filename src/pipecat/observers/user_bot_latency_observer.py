"""Observer for tracking user-to-bot response latency.

This module provides an observer that monitors the time between when a user
stops speaking and when the bot starts speaking, emitting events when latency
is measured.
"""

import time
from typing import Optional, Set

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class UserBotLatencyObserver(BaseObserver):
    """Observer that tracks user-to-bot response latency.

    Measures the time between when a user stops speaking (VADUserStoppedSpeakingFrame)
    and when the bot starts speaking (BotStartedSpeakingFrame). Emits events when
    latency is measured, allowing consumers to log, trace, or otherwise process
    the latency data.

    This observer follows the composition pattern used by TurnTrackingObserver,
    acting as a reusable component for latency measurement.

    Events:
        on_latency_measured(observer, latency_seconds): Emitted when user-to-bot
            latency is calculated. Includes the latency value in seconds as a float.
    """

    def __init__(self, **kwargs):
        """Initialize the user-bot latency observer.

        Sets up tracking for processed frames and user speech timing
        to calculate response latencies.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._user_stopped_time: Optional[float] = None
        self._processed_frames: Set[str] = set()

        self._register_event_handler("on_latency_measured")

    async def on_push_frame(self, data: FramePushed):
        """Process frames to track speech timing and calculate latency.

        Tracks VAD events and bot speaking events to measure the time between
        user stopping speech and bot starting speech.

        Args:
            data: Frame push event containing the frame and direction information.
        """
        # Only process downstream frames
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)

        # Track VAD and bot speaking events for latency
        if isinstance(data.frame, VADUserStartedSpeakingFrame):
            # Reset when user starts speaking
            self._user_stopped_time = None
        elif isinstance(data.frame, VADUserStoppedSpeakingFrame):
            # Record the actual time the user stopped speaking, which is
            # the VAD determination time minus the stop_secs silence duration
            # that had to elapse before the VAD confirmed speech ended.
            self._user_stopped_time = data.frame.timestamp - data.frame.stop_secs
        elif isinstance(data.frame, BotStartedSpeakingFrame) and self._user_stopped_time:
            # Calculate and emit latency
            latency = time.time() - self._user_stopped_time
            self._user_stopped_time = None
            await self._call_event_handler("on_latency_measured", latency)
