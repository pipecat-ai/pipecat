#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer for measuring user-to-bot response latency."""

from statistics import mean

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver


class UserBotLatencyLogObserver(BaseObserver):
    """Observer that logs user-to-bot response latency.

    Uses UserBotLatencyObserver to track latency measurements and provides
    logging and statistics. Logs individual latencies and a summary with
    average, min, and max values when the pipeline ends.
    """

    def __init__(self, latency_tracker: UserBotLatencyObserver, **kwargs):
        """Initialize the latency log observer.

        Args:
            latency_tracker: The latency tracking observer to monitor.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._latency_tracker = latency_tracker
        self._latencies = []

        if latency_tracker:

            @latency_tracker.event_handler("on_latency_measured")
            async def on_latency_measured(tracker, latency_seconds):
                await self._handle_latency_measured(latency_seconds)

    async def on_push_frame(self, data: FramePushed):
        """Process frames to handle pipeline end events.

        Args:
            data: Frame push event containing the frame and direction information.
        """
        if isinstance(data.frame, (EndFrame, CancelFrame)):
            self._log_summary()

    async def _handle_latency_measured(self, latency_seconds: float):
        """Handle latency measurement events.

        Called when the latency tracker measures user-to-bot latency.
        Stores the latency and logs it.

        Args:
            latency_seconds: The measured latency in seconds.
        """
        self._latencies.append(latency_seconds)
        self._log_latency(latency_seconds)

    def _log_summary(self):
        if not self._latencies:
            return
        avg_latency = mean(self._latencies)
        min_latency = min(self._latencies)
        max_latency = max(self._latencies)
        logger.info(
            f"⏱️ LATENCY FROM USER STOPPED SPEAKING TO BOT STARTED SPEAKING - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s"
        )

    def _log_latency(self, latency: float):
        """Log the latency.

        Args:
            latency: The latency to log.
        """
        logger.debug(
            f"⏱️ LATENCY FROM USER STOPPED SPEAKING TO BOT STARTED SPEAKING: {latency:.3f}s"
        )
