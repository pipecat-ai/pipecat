import time

from loguru import logger

from pipecat.observers.turn_tracking_observer import TurnTrackingObserver


class TurnLatencyTracker:
    """Track the latency of each turn."""

    def __init__(self):
        self._turn_start_time = None
        self._latency_history = []

    def _reset_turn_start_time(self):
        self._turn_start_time = None

    def on_turn_started(self):
        self._turn_start_time = time.time()

    def on_turn_ended(self):
        if self._turn_start_time:
            latency = time.time() - self._turn_start_time
            self._latency_history.append(latency)
            logger.info(f"Turn ended with latency {latency} seconds")
            self._reset_turn_start_time()

    def get_latency_stats(self):
        return {
            "average": sum(self._latency_history) / len(self._latency_history),
            "median": sorted(self._latency_history)[len(self._latency_history) // 2],
            "min": min(self._latency_history),
            "max": max(self._latency_history),
        }
