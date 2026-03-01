#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer for tracking user-to-bot response latency.

This module provides an observer that monitors the time between when a user
stops speaking and when the bot starts speaking, emitting events when latency
is measured. Optionally collects per-service latency breakdown metrics
(TTFB, text aggregation) when ``enable_metrics=True``.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ClientConnectedFrame,
    InterruptionFrame,
    MetricsFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    TextAggregationMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


@dataclass
class LatencyBreakdown:
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.task.PipelineParams`.

    Parameters:
        ttfb: Time-to-first-byte metrics from each service in the pipeline.
        text_aggregation: First text aggregation measurement, representing
            the latency cost of sentence aggregation in the TTS pipeline.
        user_turn_secs: Duration in seconds of the user's turn, measured
            from when the user actually stopped speaking to when the turn
            was released (``UserStoppedSpeakingFrame``). This includes
            VAD silence detection, STT finalization, and any turn analyzer
            wait. ``None`` if no ``UserStoppedSpeakingFrame`` was observed
            (e.g. no turn analyzer configured).
    """

    ttfb: List[TTFBMetricsData] = field(default_factory=list)
    text_aggregation: Optional[TextAggregationMetricsData] = None
    user_turn_secs: Optional[float] = None


class UserBotLatencyObserver(BaseObserver):
    """Observer that tracks user-to-bot response latency.

    Measures the time between when a user stops speaking (VADUserStoppedSpeakingFrame)
    and when the bot starts speaking (BotStartedSpeakingFrame). Emits events when
    latency is measured, allowing consumers to log, trace, or otherwise process
    the latency data.

    When ``enable_metrics=True`` in pipeline params, also collects per-service
    latency breakdown (TTFB, text aggregation) and emits an
    ``on_latency_breakdown`` event alongside the existing latency measurement.

    This observer follows the composition pattern used by TurnTrackingObserver,
    acting as a reusable component for latency measurement.

    Events:
        on_latency_measured(observer, latency_seconds): Emitted when
            time-to-first-bot-speech is calculated. Measures the time from
            when the user stopped speaking to when the bot starts speaking.
        on_latency_breakdown(observer, breakdown): Emitted at each
            ``BotStartedSpeakingFrame`` with a :class:`LatencyBreakdown`
            containing per-service metrics collected during the user→bot cycle.
        on_first_bot_speech_latency(observer, latency_seconds): Emitted once,
            the first time ``BotStartedSpeakingFrame`` arrives after
            ``ClientConnectedFrame``. Measures the time from client connection
            to the first bot speech.
    """

    def __init__(self, *, max_frames=100, **kwargs):
        """Initialize the user-bot latency observer.

        Sets up tracking for processed frames and user speech timing
        to calculate response latencies.

        Args:
            max_frames: Maximum number of frame IDs to keep in history for
                duplicate detection. Defaults to 100.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._user_stopped_time: Optional[float] = None
        self._user_turn: Optional[float] = None

        # First bot speech tracking
        self._client_connected_time: Optional[float] = None
        self._first_bot_speech_measured: bool = False

        # Frame deduplication (bounded deque + set pattern)
        self._processed_frames: set = set()
        self._frame_history: deque = deque(maxlen=max_frames)

        # Per-cycle metric accumulators
        self._ttfb: List[TTFBMetricsData] = []
        self._text_aggregation: Optional[TextAggregationMetricsData] = None

        self._register_event_handler("on_latency_measured")
        self._register_event_handler("on_latency_breakdown")
        self._register_event_handler("on_first_bot_speech_latency")

    async def on_push_frame(self, data: FramePushed):
        """Process frames to track speech timing and calculate latency.

        Tracks VAD events and bot speaking events to measure the time between
        user stopping speech and bot starting speech. Also accumulates metrics
        from MetricsFrame for the latency breakdown.

        Args:
            data: Frame push event containing the frame and direction information.
        """
        # Only process downstream frames
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # Skip already processed frames (bounded deque + set)
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)
        self._frame_history.append(data.frame.id)

        if len(self._processed_frames) > len(self._frame_history):
            self._processed_frames = set(self._frame_history)

        # Track client connection (first occurrence only)
        if isinstance(data.frame, ClientConnectedFrame):
            if self._client_connected_time is None:
                self._client_connected_time = time.time()
            return

        # Track speech and pipeline events for latency
        if isinstance(data.frame, VADUserStartedSpeakingFrame):
            # Reset when user starts speaking
            self._user_stopped_time = None
            self._user_turn = None
            self._reset_accumulators()
            # If user speaks before the bot's first speech, abandon the
            # first-bot-speech measurement — it's only meaningful for greetings.
            self._first_bot_speech_measured = True
        elif isinstance(data.frame, VADUserStoppedSpeakingFrame):
            # Record the actual time the user stopped speaking, which is
            # the VAD determination time minus the stop_secs silence duration
            # that had to elapse before the VAD confirmed speech ended.
            self._user_stopped_time = data.frame.timestamp - data.frame.stop_secs
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            # Measure the user turn duration: from actual user silence to
            # turn release. Includes VAD silence detection, STT finalization,
            # and any turn analyzer wait.
            if self._user_stopped_time is not None:
                self._user_turn = time.time() - self._user_stopped_time
        elif isinstance(data.frame, InterruptionFrame):
            # Discard stale metrics from cancelled LLM/TTS cycles
            self._reset_accumulators()
        elif isinstance(data.frame, MetricsFrame):
            self._handle_metrics_frame(data.frame)
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking()

    async def _handle_bot_started_speaking(self):
        """Handle BotStartedSpeakingFrame to emit latency and breakdown."""
        emit_breakdown = False

        # One-time first bot speech measurement (client connect → first speech)
        if self._client_connected_time is not None and not self._first_bot_speech_measured:
            self._first_bot_speech_measured = True
            latency = time.time() - self._client_connected_time
            await self._call_event_handler("on_first_bot_speech_latency", latency)
            emit_breakdown = True

        if self._user_stopped_time is not None:
            latency = time.time() - self._user_stopped_time
            self._user_stopped_time = None
            await self._call_event_handler("on_latency_measured", latency)
            emit_breakdown = True

        if emit_breakdown:
            breakdown = LatencyBreakdown(
                ttfb=list(self._ttfb),
                text_aggregation=self._text_aggregation,
                user_turn_secs=self._user_turn,
            )
            await self._call_event_handler("on_latency_breakdown", breakdown)
            self._reset_accumulators()

    def _handle_metrics_frame(self, frame: MetricsFrame):
        """Extract latency metrics from a MetricsFrame.

        Accumulates metrics when a measurement is in progress: either a
        user→bot cycle (after ``VADUserStoppedSpeakingFrame``) or the
        first-bot-speech window (after ``ClientConnectedFrame``).
        """
        waiting_for_first_speech = (
            self._client_connected_time is not None and not self._first_bot_speech_measured
        )
        if self._user_stopped_time is None and not waiting_for_first_speech:
            return

        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData) and metrics_data.value > 0:
                self._ttfb.append(metrics_data)
            elif isinstance(metrics_data, TextAggregationMetricsData):
                # Only keep the first measurement — it's the one that
                # impacts the initial speaking latency.
                if self._text_aggregation is None:
                    self._text_aggregation = metrics_data

    def _reset_accumulators(self):
        """Clear per-cycle metric accumulators."""
        self._ttfb = []
        self._text_aggregation = None
        self._user_turn = None
