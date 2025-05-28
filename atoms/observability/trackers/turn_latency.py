import time

from loguru import logger

from pipecat.observers.turn_tracking_observer import TurnTrackingObserver


class TurnLatencyTracker:
    """Track the latency of each turn."""

    def __init__(self, turn_tracking_observer: TurnTrackingObserver):
        self._turn_tracking_observer = turn_tracking_observer
        self._turn_start_time = None
        self._latency_history = []

        on_turn_started = self._on_turn_started
        on_turn_ended = self._on_turn_ended

        @self._turn_tracking_observer.event_handler("on_turn_started")
        async def on_turn_started(self, turn_count: int):
            on_turn_started()

        @self._turn_tracking_observer.event_handler("on_turn_ended")
        async def on_turn_ended(self, turn_count: int, duration: float, was_interrupted: bool):
            on_turn_ended()

    def _reset_turn_start_time(self):
        self._turn_start_time = None

    def _on_turn_started(self):
        self._turn_start_time = time.time()

    def _on_turn_ended(self):
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
