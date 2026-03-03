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
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ClientConnectedFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
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


class TTFBBreakdownMetrics(BaseModel):
    """TTFB measurement with timestamp for timeline placement.

    Parameters:
        processor: Name of the processor that reported the TTFB.
        model: Optional model name associated with the metric.
        start_time: Unix timestamp when the TTFB measurement started.
        duration_secs: TTFB duration in seconds.
    """

    processor: str
    model: Optional[str] = None
    start_time: float
    duration_secs: float


class TextAggregationBreakdownMetrics(BaseModel):
    """Text aggregation measurement with timestamp for timeline placement.

    Parameters:
        processor: Name of the processor that reported the metric.
        start_time: Unix timestamp when text aggregation started.
        duration_secs: Aggregation duration in seconds.
    """

    processor: str
    start_time: float
    duration_secs: float


class FunctionCallMetrics(BaseModel):
    """Latency for a single function call execution.

    Parameters:
        function_name: Name of the function that was called.
        start_time: Unix timestamp when execution started.
        duration_secs: Time in seconds from execution start to result.
    """

    function_name: str
    start_time: float
    duration_secs: float


class LatencyBreakdown(BaseModel):
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.task.PipelineParams`.

    Parameters:
        ttfb: Time-to-first-byte metrics from each service in the pipeline.
        text_aggregation: First text aggregation measurement, representing
            the latency cost of sentence aggregation in the TTS pipeline.
        user_turn_start_time: Unix timestamp when the user turn started
            (actual user silence, adjusted for VAD stop_secs). ``None`` if
            no ``VADUserStoppedSpeakingFrame`` was observed.
        user_turn_secs: Duration in seconds of the user's turn, measured
            from when the user actually stopped speaking to when the turn
            was released (``UserStoppedSpeakingFrame``). This includes
            VAD silence detection, STT finalization, and any turn analyzer
            wait. ``None`` if no ``UserStoppedSpeakingFrame`` was observed
            (e.g. no turn analyzer configured).
        function_calls: Latency for each function call executed during
            this cycle. Empty if no function calls occurred.
    """

    ttfb: List[TTFBBreakdownMetrics] = Field(default_factory=list)
    text_aggregation: Optional[TextAggregationBreakdownMetrics] = None
    user_turn_start_time: Optional[float] = None
    user_turn_secs: Optional[float] = None
    function_calls: List[FunctionCallMetrics] = Field(default_factory=list)

    def chronological_events(self) -> List[str]:
        """Return human-readable event labels sorted by start time.

        Collects all sub-metrics into a flat list, sorts by ``start_time``,
        and returns formatted strings suitable for logging.

        Returns:
            List of formatted strings, one per event, in chronological order.
        """
        events: List[tuple] = []

        if self.user_turn_start_time is not None and self.user_turn_secs is not None:
            events.append((self.user_turn_start_time, f"User turn: {self.user_turn_secs:.3f}s"))

        for t in self.ttfb:
            events.append((t.start_time, f"{t.processor}: TTFB {t.duration_secs:.3f}s"))

        for fc in self.function_calls:
            events.append((fc.start_time, f"{fc.function_name}: {fc.duration_secs:.3f}s"))

        if self.text_aggregation:
            ta = self.text_aggregation
            events.append(
                (ta.start_time, f"{ta.processor}: text aggregation {ta.duration_secs:.3f}s")
            )

        events.sort(key=lambda e: e[0])
        return [label for _, label in events]


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
        self._user_turn_start_time: Optional[float] = None
        self._user_turn: Optional[float] = None

        # First bot speech tracking
        self._client_connected_time: Optional[float] = None
        self._first_bot_speech_measured: bool = False

        # Frame deduplication (bounded deque + set pattern)
        self._processed_frames: set = set()
        self._frame_history: deque = deque(maxlen=max_frames)

        # Per-cycle metric accumulators
        self._ttfb: List[TTFBBreakdownMetrics] = []
        self._text_aggregation: Optional[TextAggregationBreakdownMetrics] = None
        self._function_call_starts: Dict[str, tuple[str, float]] = {}
        self._function_call_metrics: List[FunctionCallMetrics] = []

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
            self._user_turn_start_time = None
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
            self._user_turn_start_time = self._user_stopped_time
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            # Measure the user turn duration: from actual user silence to
            # turn release. Includes VAD silence detection, STT finalization,
            # and any turn analyzer wait.
            if self._user_stopped_time is not None:
                self._user_turn = time.time() - self._user_stopped_time
        elif isinstance(data.frame, InterruptionFrame):
            # Discard stale metrics from cancelled LLM/TTS cycles
            self._reset_accumulators()
        elif isinstance(data.frame, FunctionCallInProgressFrame):
            self._function_call_starts[data.frame.tool_call_id] = (
                data.frame.function_name,
                time.time(),
            )
        elif isinstance(data.frame, FunctionCallResultFrame):
            start = self._function_call_starts.pop(data.frame.tool_call_id, None)
            if start is not None:
                function_name, start_time = start
                self._function_call_metrics.append(
                    FunctionCallMetrics(
                        function_name=function_name,
                        start_time=start_time,
                        duration_secs=time.time() - start_time,
                    )
                )
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
                user_turn_start_time=self._user_turn_start_time,
                user_turn_secs=self._user_turn,
                function_calls=list(self._function_call_metrics),
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

        now = time.time()
        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData) and metrics_data.value > 0:
                self._ttfb.append(
                    TTFBBreakdownMetrics(
                        processor=metrics_data.processor,
                        model=metrics_data.model,
                        start_time=now - metrics_data.value,
                        duration_secs=metrics_data.value,
                    )
                )
            elif isinstance(metrics_data, TextAggregationMetricsData):
                # Only keep the first measurement — it's the one that
                # impacts the initial speaking latency.
                if self._text_aggregation is None:
                    self._text_aggregation = TextAggregationBreakdownMetrics(
                        processor=metrics_data.processor,
                        start_time=now - metrics_data.value,
                        duration_secs=metrics_data.value,
                    )

    def _reset_accumulators(self):
        """Clear per-cycle metric accumulators."""
        self._ttfb = []
        self._text_aggregation = None
        self._user_turn_start_time = None
        self._user_turn = None
        self._function_call_starts = {}
        self._function_call_metrics = []
