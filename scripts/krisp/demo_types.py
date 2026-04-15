"""Shared data types for the turn-taking demo tool."""

from dataclasses import dataclass, field
from typing import List, Optional

METHOD_STREAMING = "streaming"
METHOD_ON_DEMAND = "on-demand"
METHOD_TIMEOUT = "timeout"


@dataclass
class TurnEvent:
    """A detected turn-complete event."""

    timestamp: float
    silence_start: Optional[float] = None
    method: str = METHOD_STREAMING

    @property
    def detection_delay(self) -> Optional[float]:
        """Time from VAD-declared-silence to turn detection."""
        if self.silence_start is not None:
            return self.timestamp - self.silence_start
        return None


@dataclass
class AnalyzerResult:
    """Collected results for one analyzer run."""

    name: str
    turn_events: List[TurnEvent] = field(default_factory=list)
    init_time_ms: float = 0.0
    timeout_secs: Optional[float] = None
