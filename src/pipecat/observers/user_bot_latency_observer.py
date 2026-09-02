#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer for tracking user-to-bot response latency.

This module provides an observer that monitors the time between when a user
stops speaking and when the bot starts speaking, emitting events when latency
is measured. When ``enable_metrics=True`` it also collects a breakdown of the
interval: per-service metrics (TTFB, text aggregation), and contributions that
name each part of the timeline, including the parts no service measures.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto

from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ClientConnectedFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
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
from pipecat.utils.deprecation import deprecated


class TTFBBreakdownMetrics(BaseModel):
    """TTFB measurement with timestamp for timeline placement.

    Parameters:
        processor: Name of the processor that reported the TTFB.
        model: Optional model name associated with the metric.
        start_time: Unix timestamp when the TTFB measurement started.
        duration_secs: TTFB duration in seconds.
    """

    processor: str
    model: str | None = None
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


# Anything shorter than this is a frame hop rather than work worth naming, so
# it is rolled into the single pipeline contribution.
MIN_CONTRIBUTION_SECS = 0.005

# Half of the last digit contribution_lines() prints, below which a span would
# show as zero and so is not part of the timeline at all.
PRINTS_AS_ZERO_SECS = 0.0005


class _MomentKind(StrEnum):
    """A point in a user-to-bot cycle that a contribution can start or end at.

    Most are observed as frames. ``LLM_CHUNK`` and ``SENTENCE`` are worked back
    from a service's own metric, because no frame marks them.
    """

    SILENCE = "silence"
    VAD_STOP = "vad stop"
    TRANSCRIPT = "transcript"
    LLM_REQUEST = "llm request"
    LLM_CHUNK = "llm chunk"
    MARKER_COMPLETE = "marker complete"
    MARKER_INCOMPLETE = "marker incomplete"
    HANDLERS_START = "handlers start"
    HANDLERS_END = "handlers end"
    FIRST_TEXT = "first text"
    SENTENCE = "sentence"
    FIRST_AUDIO = "first audio"
    BOT_SPEAKING = "bot speaking"


@dataclass(frozen=True)
class _Moment:
    """One moment in a cycle.

    Parameters:
        at: Unix timestamp of the moment.
        kind: What happened.
        source: Name of whatever produced it, used when a span takes its owner
            from one end.
        derived: Whether it was worked back from a metric rather than observed
            as a frame. A derived moment sits at or before the frame that
            reported it, which decides the order when the two share a time.
    """

    at: float
    kind: _MomentKind
    source: str = ""
    derived: bool = False


class _Owner(Enum):
    """Which end of a span supplies its owner, when a setting does not."""

    OPENER = auto()
    CLOSER = auto()


_TURN_COMPLETION = "config: filter_incomplete_user_turns"

# Each pair of adjacent moments names the span between them, and a pair that is
# absent leaves that stretch unnamed, to be reported as pipeline time. A setting
# that governs the time owns it; otherwise the owner is the processor at one
# end. The flag marks a core stage, which stays listed however brief so the
# timeline keeps its shape from one turn to the next.
_SPANS: dict[tuple[_MomentKind, _MomentKind], tuple[str, str | _Owner, bool]] = {
    # Silence the VAD had to hear before it would call the turn over. It is a
    # setting rather than a service, and shortening it trades latency for
    # transcription accuracy.
    (_MomentKind.SILENCE, _MomentKind.VAD_STOP): (
        "endpointing wait",
        "config: VAD stop_secs",
        True,
    ),
    (_MomentKind.VAD_STOP, _MomentKind.TRANSCRIPT): ("transcription", _Owner.CLOSER, True),
    # Whatever decides the user is done: a fixed speech timeout, a smart-turn
    # model's inference, or an external signal.
    (_MomentKind.TRANSCRIPT, _MomentKind.LLM_REQUEST): (
        "turn detection",
        "config: user turn strategies",
        True,
    ),
    (_MomentKind.LLM_REQUEST, _MomentKind.LLM_CHUNK): ("LLM inference", _Owner.OPENER, True),
    # The gate reading the verdict, whichever way it went.
    (_MomentKind.LLM_CHUNK, _MomentKind.MARKER_COMPLETE): (
        "turn completion",
        _TURN_COMPLETION,
        False,
    ),
    (_MomentKind.LLM_CHUNK, _MomentKind.MARKER_INCOMPLETE): (
        "turn completion",
        _TURN_COMPLETION,
        False,
    ),
    # The token the marker occupies before anything can be spoken.
    (_MomentKind.MARKER_COMPLETE, _MomentKind.FIRST_TEXT): (
        "turn completion",
        _TURN_COMPLETION,
        False,
    ),
    (_MomentKind.MARKER_INCOMPLETE, _MomentKind.LLM_REQUEST): (
        "waiting for user",
        _TURN_COMPLETION,
        False,
    ),
    # Between the LLM's first chunk and the call being dispatched, the LLM is
    # still writing the call.
    (_MomentKind.LLM_CHUNK, _MomentKind.HANDLERS_START): ("LLM tool call", _Owner.OPENER, False),
    (_MomentKind.HANDLERS_START, _MomentKind.HANDLERS_END): (
        "function handler",
        _Owner.CLOSER,
        False,
    ),
    # Waiting for a full sentence before speaking any of it.
    (_MomentKind.FIRST_TEXT, _MomentKind.SENTENCE): (
        "sentence aggregation",
        "config: text_aggregation_mode",
        False,
    ),
    (_MomentKind.SENTENCE, _MomentKind.FIRST_AUDIO): ("speech synthesis", _Owner.CLOSER, True),
    (_MomentKind.FIRST_TEXT, _MomentKind.FIRST_AUDIO): ("speech synthesis", _Owner.CLOSER, True),
    (_MomentKind.FIRST_AUDIO, _MomentKind.BOT_SPEAKING): ("output transport", _Owner.CLOSER, False),
}


class LatencyContribution(BaseModel):
    """One named part of the user-to-bot interval.

    Contributions account for the whole interval, so their durations sum to the
    measured latency. Unlike a TTFB, which reports what a service spent once it
    was asked, a contribution can also name time no service is measuring: the
    silence a VAD waits out, the tokens a turn-completion marker occupies before
    any speakable text, or a hold while the pipeline waits for the user to
    finish a sentence.

    Parameters:
        label: What the time was spent on.
        owner: What spent it — a processor name, or a ``config:`` tag naming
            the setting that governs it.
        start_time: Unix timestamp when it started.
        duration_secs: How long it took, in seconds.
    """

    label: str
    owner: str
    start_time: float
    duration_secs: float


class LatencyBreakdown(BaseModel):
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.worker.PipelineParams`.

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
        contributions: Named parts of the interval, in chronological order,
            summing to the measured latency. A part is listed only if it
            happened, so a bot without turn completion has no marker or wait
            entries.
    """

    ttfb: list[TTFBBreakdownMetrics] = Field(default_factory=list)
    contributions: list[LatencyContribution] = Field(default_factory=list)
    text_aggregation: TextAggregationBreakdownMetrics | None = None
    user_turn_start_time: float | None = None
    user_turn_secs: float | None = None
    function_calls: list[FunctionCallMetrics] = Field(default_factory=list)

    @deprecated(
        "`LatencyBreakdown.chronological_events` is deprecated since 1.9.0 and will be removed "
        "in 2.0.0. Use `LatencyBreakdown.contribution_lines` instead."
    )
    def chronological_events(self) -> list[str]:
        """Return human-readable event labels sorted by start time.

        .. deprecated:: 1.9.0
            Use :meth:`contribution_lines` instead, which names every part of
            the interval rather than the services that reported a metric. Will
            be removed in 2.0.0.

        Collects all sub-metrics into a flat list, sorts by ``start_time``,
        and returns formatted strings suitable for logging.

        Returns:
            List of formatted strings, one per event, in chronological order.
        """
        events: list[tuple] = []

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

    def contribution_lines(self, *, by_cost: bool = False) -> list[str]:
        """Format the contributions for logging, one per line plus a total.

        Args:
            by_cost: Order by duration, largest first, rather than in the
                order things happened. Useful when the question is what to
                optimize rather than what the turn did.

        Returns:
            One formatted line per contribution, then a total line. Empty if
            no contributions were collected.
        """
        if not self.contributions:
            return []

        ordered = (
            sorted(self.contributions, key=lambda c: -c.duration_secs)
            if by_cost
            else self.contributions
        )
        lines = [f"{c.duration_secs:6.3f}s  {c.label:20} [{c.owner}]" for c in ordered]
        lines.append(f"{sum(c.duration_secs for c in self.contributions):6.3f}s  TOTAL")
        return lines


class UserBotLatencyObserver(BaseObserver):
    """Observer that tracks user-to-bot response latency.

    Measures the time between when a user stops speaking (VADUserStoppedSpeakingFrame)
    and when the bot starts speaking (BotStartedSpeakingFrame). Emits events when
    latency is measured, allowing consumers to log, trace, or otherwise process
    the latency data.

    When ``enable_metrics=True`` in pipeline params, also collects a latency
    breakdown and emits an ``on_latency_breakdown`` event alongside the
    existing latency measurement. The breakdown carries per-service metrics
    (TTFB, text aggregation) and, in ``contributions``, the whole interval
    named part by part::

        for line in breakdown.contribution_lines():
            logger.info(line)

        # 0.200s  endpointing wait     [config: VAD stop_secs]
        # 0.125s  transcription        [DeepgramSTTService#0]
        # 0.336s  LLM inference        [OpenAILLMService#0]
        # 0.359s  speech synthesis     [CartesiaTTSService#0]
        # 1.020s  TOTAL

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

    def __init__(
        self,
        *,
        max_frames=100,
        min_contribution_secs: float = MIN_CONTRIBUTION_SECS,
        time_source: Callable[[], float] = time.time,
        **kwargs,
    ):
        """Initialize the user-bot latency observer.

        Sets up tracking for processed frames and user speech timing
        to calculate response latencies.

        Args:
            max_frames: Maximum number of frame IDs to keep in history for
                duplicate detection. Defaults to 100.
            min_contribution_secs: Contributions shorter than this are rolled
                into the single pipeline entry rather than listed. Pass 0 to
                list every one, including individual frame hops.
            time_source: Reads the current time in seconds. Supplying one lets
                a test drive a cycle without waiting out the intervals it
                describes.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_contribution_secs = min_contribution_secs
        self._now = time_source
        self._user_stopped_time: float | None = None
        self._user_turn_start_time: float | None = None
        self._user_turn: float | None = None

        # First bot speech tracking
        self._client_connected_time: float | None = None
        self._first_bot_speech_measured: bool = False

        # Frame deduplication (bounded deque + set pattern)
        self._processed_frames: set = set()
        self._frame_history: deque = deque(maxlen=max_frames)

        # The moments of the cycle, in the order they were observed.
        self._moments: list[_Moment] = []
        self._llm_request: _Moment | None = None
        # Whether this bot uses turn completion at all, which a marker frame
        # proves. Not per-cycle: a bot that emits markers keeps using them.
        self._markers_seen: bool = False

        # Per-cycle metric accumulators
        self._ttfb: list[TTFBBreakdownMetrics] = []
        self._text_aggregation: TextAggregationBreakdownMetrics | None = None
        self._function_call_starts: dict[str, tuple[str, float]] = {}
        self._function_call_metrics: list[FunctionCallMetrics] = []

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
                self._client_connected_time = self._now()
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
            self._mark(_MomentKind.VAD_STOP, at=data.frame.timestamp)
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            # Measure the user turn duration: from actual user silence to
            # turn release. Includes VAD silence detection, STT finalization,
            # and any turn analyzer wait.
            if self._user_stopped_time is not None:
                self._user_turn = self._now() - self._user_stopped_time
        elif isinstance(data.frame, TranscriptionFrame):
            # A service can finalize an utterance in several pieces. The turn is
            # not detectable until the last of them, which is also where the
            # service stops its own TTFB clock, so a later one replaces the
            # earlier until the LLM is asked.
            if not self._seen(_MomentKind.LLM_REQUEST):
                self._moments = [m for m in self._moments if m.kind is not _MomentKind.TRANSCRIPT]
                self._mark(_MomentKind.TRANSCRIPT, source=data.source.name)
        elif isinstance(data.frame, LLMFullResponseStartFrame):
            self._llm_request = self._mark(_MomentKind.LLM_REQUEST, source=data.source.name)
        elif isinstance(data.frame, LLMMarkerFrame):
            # A stand-alone marker holds the turn open; one that prefixes a
            # response is the completion the pipeline was waiting for.
            self._markers_seen = True
            # A turn can be held more than once before it completes, so every
            # incomplete verdict is recorded; only the completion is kept once.
            self._mark(
                _MomentKind.MARKER_INCOMPLETE
                if data.frame.append_to_context_immediately
                else _MomentKind.MARKER_COMPLETE,
                once=not data.frame.append_to_context_immediately,
            )
        elif isinstance(data.frame, LLMTextFrame):
            self._mark(_MomentKind.FIRST_TEXT, source=data.source.name, once=True)
        elif isinstance(data.frame, TTSAudioRawFrame):
            self._mark(_MomentKind.FIRST_AUDIO, source=data.source.name, once=True)
        elif isinstance(data.frame, InterruptionFrame):
            # Discard stale metrics from cancelled LLM/TTS cycles
            self._reset_accumulators()
        elif isinstance(data.frame, FunctionCallInProgressFrame):
            self._function_call_starts[data.frame.tool_call_id] = (
                data.frame.function_name,
                self._now(),
            )
        elif isinstance(data.frame, FunctionCallResultFrame):
            start = self._function_call_starts.pop(data.frame.tool_call_id, None)
            if start is not None:
                function_name, start_time = start
                self._function_call_metrics.append(
                    FunctionCallMetrics(
                        function_name=function_name,
                        start_time=start_time,
                        duration_secs=self._now() - start_time,
                    )
                )
        elif isinstance(data.frame, MetricsFrame):
            self._handle_metrics_frame(data.frame)
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            self._mark(_MomentKind.BOT_SPEAKING, source=data.source.name, once=True)
            await self._handle_bot_started_speaking()

    def _mark(
        self,
        kind: _MomentKind,
        *,
        at: float | None = None,
        source: str = "",
        once: bool = False,
        derived: bool = False,
    ) -> _Moment | None:
        """Record a moment, if a cycle is being measured.

        Args:
            kind: What happened.
            at: When, defaulting to now. Derived moments pass their own time.
            source: Name of whatever produced it.
            once: Keep only the first of this kind, for frames that repeat
                within a cycle such as text and audio.
            derived: Whether it comes from a metric rather than from a frame.

        Returns:
            The recorded moment, or None if it was not recorded.
        """
        if self._user_stopped_time is None:
            return None
        if once and self._seen(kind):
            return None
        moment = _Moment(
            at=at if at is not None else self._now(), kind=kind, source=source, derived=derived
        )
        self._moments.append(moment)
        return moment

    def _seen(self, kind: _MomentKind) -> bool:
        """Whether a moment of this kind has been recorded this cycle."""
        return any(m.kind is kind for m in self._moments)

    def _derived_moments(self) -> list[_Moment]:
        """Moments that come from metrics rather than from frames.

        Handlers dispatched together run concurrently, so they become one pair
        of moments named for the one the wait actually depends on: listing each
        would count the same stretch of wall clock more than once.

        Returns:
            The sentence-aggregation and function-handler moments, if any.
        """
        moments: list[_Moment] = []

        first_text = next((m for m in self._moments if m.kind is _MomentKind.FIRST_TEXT), None)
        if first_text and self._text_aggregation and self._text_aggregation.duration_secs > 0:
            moments.append(
                _Moment(
                    at=first_text.at + self._text_aggregation.duration_secs,
                    kind=_MomentKind.SENTENCE,
                    derived=True,
                )
            )

        for group in self._concurrent_handlers():
            slowest = max(group, key=lambda c: c.duration_secs)
            owner = slowest.function_name
            if len(group) > 1:
                owner += f" +{len(group) - 1}"
            moments.append(
                _Moment(
                    at=min(c.start_time for c in group),
                    kind=_MomentKind.HANDLERS_START,
                    derived=True,
                )
            )
            moments.append(
                _Moment(
                    at=max(c.start_time + c.duration_secs for c in group),
                    kind=_MomentKind.HANDLERS_END,
                    source=owner,
                    derived=True,
                )
            )
        return moments

    def _concurrent_handlers(self) -> list[list[FunctionCallMetrics]]:
        """Group function handlers whose executions overlap in time.

        Returns:
            Groups of overlapping calls, in the order they started.
        """
        groups: list[list[FunctionCallMetrics]] = []
        for call in sorted(self._function_call_metrics, key=lambda c: c.start_time):
            if groups and call.start_time <= max(
                c.start_time + c.duration_secs for c in groups[-1]
            ):
                groups[-1].append(call)
            else:
                groups.append([call])
        return groups

    def _build_contributions(self) -> list[LatencyContribution]:
        """Name each stretch between one moment and the next.

        Spans run from one moment to the following one, so they tile the cycle
        and cannot overlap. A stretch whose pair of moments has no entry in the
        table is left unnamed and reported as pipeline time, which is what makes
        a gap in the accounting visible rather than silently absorbed.

        Returns:
            Contributions in chronological order, or empty if the cycle was
            never measured.
        """
        start = self._user_turn_start_time
        if start is None or not self._seen(_MomentKind.BOT_SPEAKING):
            return []

        moments = sorted(
            [_Moment(at=start, kind=_MomentKind.SILENCE), *self._moments, *self._derived_moments()],
            key=lambda m: (m.at, not m.derived),
        )

        spans: list[LatencyContribution] = []
        core: set[str] = set()
        for opener, closer in zip(moments, moments[1:]):
            if closer.at <= opener.at:
                continue
            if (opener.kind, closer.kind) == (_MomentKind.LLM_CHUNK, _MomentKind.FIRST_TEXT):
                # Where turn completion is in use, a response that carried no
                # marker is buffered whole before anything can be spoken. Where
                # it is not, the same wait is the LLM still streaming, so its
                # inference covers it.
                entry = (
                    ("awaiting speakable text", _Owner.CLOSER, False)
                    if self._markers_seen
                    else ("LLM inference", _Owner.OPENER, True)
                )
            else:
                entry = _SPANS.get((opener.kind, closer.kind))
            if entry is None:
                continue
            label, owner, is_core = entry
            if owner is _Owner.OPENER:
                owner = opener.source
            elif owner is _Owner.CLOSER:
                owner = closer.source
            if is_core:
                core.add(label)
            spans.append(
                LatencyContribution(
                    label=label,
                    owner=owner,
                    start_time=opener.at,
                    duration_secs=closer.at - opener.at,
                )
            )

        spans = self._merge_adjacent(spans)

        # A core stage stays listed however brief, but nothing is listed that
        # would print as zero: a stage that took no measurable time is not part
        # of the timeline.
        named = [
            c
            for c in spans
            if c.duration_secs >= self._min_contribution_secs
            or (c.label in core and c.duration_secs >= PRINTS_AS_ZERO_SECS)
        ]

        # Time the named stages did not cover, which includes the spans the
        # threshold filtered out. It is spread across the cycle rather than
        # sitting anywhere in it, so it is listed last, and it is listed
        # whenever there is any of it, so the contributions always sum to the
        # interval they describe.
        pipeline_secs = (moments[-1].at - start) - sum(c.duration_secs for c in named)
        if pipeline_secs >= PRINTS_AS_ZERO_SECS:
            named.append(
                LatencyContribution(
                    label="pipeline",
                    owner="pipecat",
                    start_time=start,
                    duration_secs=pipeline_secs,
                )
            )
        return named

    @staticmethod
    def _merge_adjacent(spans: list[LatencyContribution]) -> list[LatencyContribution]:
        """Join neighbouring spans that say the same thing.

        A marker sits in the middle of what a reader thinks of as one stretch,
        so the table names both halves the same way and they are joined here.
        Only spans that meet are joined: two of the same name either side of an
        unnamed gap stay apart, so the gap is reported rather than absorbed.

        Args:
            spans: Contributions in chronological order.

        Returns:
            The same coverage, with runs of one label and owner joined.
        """
        merged: list[LatencyContribution] = []
        for span in spans:
            previous = merged[-1] if merged else None
            touching = previous is not None and (
                span.start_time - (previous.start_time + previous.duration_secs)
                < PRINTS_AS_ZERO_SECS
            )
            if (
                previous
                and touching
                and (previous.label, previous.owner)
                == (
                    span.label,
                    span.owner,
                )
            ):
                previous.duration_secs = (
                    span.start_time + span.duration_secs
                ) - previous.start_time
            else:
                merged.append(span)
        return merged

    async def _handle_bot_started_speaking(self):
        """Handle BotStartedSpeakingFrame to emit latency and breakdown."""
        emit_breakdown = False

        # One-time first bot speech measurement (client connect → first speech)
        if self._client_connected_time is not None and not self._first_bot_speech_measured:
            self._first_bot_speech_measured = True
            latency = self._now() - self._client_connected_time
            await self._call_event_handler("on_first_bot_speech_latency", latency)
            emit_breakdown = True

        if self._user_stopped_time is not None:
            latency = self._now() - self._user_stopped_time
            self._user_stopped_time = None
            await self._call_event_handler("on_latency_measured", latency)
            emit_breakdown = True

        if emit_breakdown:
            breakdown = LatencyBreakdown(
                ttfb=list(self._ttfb),
                contributions=self._build_contributions(),
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

        now = self._now()
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
                if self._llm_request and metrics_data.processor == self._llm_request.source:
                    # The first chunk landed before this metric was pushed, so
                    # cap it there rather than letting a derived moment sort
                    # after the frames that followed it.
                    self._mark(
                        _MomentKind.LLM_CHUNK,
                        at=min(self._llm_request.at + metrics_data.value, now),
                        source=self._llm_request.source,
                        derived=True,
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
        self._moments = []
        self._llm_request = None
        self._ttfb = []
        self._text_aggregation = None
        self._user_turn_start_time = None
        self._user_turn = None
        self._function_call_starts = {}
        self._function_call_metrics = []
