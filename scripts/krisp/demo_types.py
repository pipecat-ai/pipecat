"""Shared data types for the turn-taking demo tool."""

from dataclasses import dataclass, field

METHOD_STREAMING = "streaming"
METHOD_ON_DEMAND = "on-demand"
METHOD_TIMEOUT = "timeout"


@dataclass
class TurnEvent:
    """A detected turn-complete event."""

    timestamp: float
    silence_start: float | None = None
    method: str = METHOD_STREAMING
    vad_stop_secs: float | None = None

    @property
    def detection_delay(self) -> float | None:
        """Time from VAD-declared-silence to turn detection."""
        if self.silence_start is not None:
            return self.timestamp - self.silence_start
        return None

    @property
    def total_delay(self) -> float | None:
        """Total latency from estimated actual speech end to turn detection.

        For SmartTurn (on-demand/timeout), includes the VAD stop delay
        that elapsed before the analyzer was invoked. For streaming
        analyzers, same as detection_delay.
        """
        d = self.detection_delay
        if d is None:
            return None
        if self.vad_stop_secs is not None:
            return d + self.vad_stop_secs
        return d


@dataclass
class AnalyzerResult:
    """Collected results for one analyzer run."""

    name: str
    turn_events: list[TurnEvent] = field(default_factory=list)
    init_time_ms: float = 0.0
    timeout_secs: float | None = None
