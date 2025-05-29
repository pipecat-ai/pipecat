import time
from collections import deque

from loguru import logger

from pipecat.frames.frames import LLMFullResponseStartFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed


class AgentResponseObserver(BaseObserver):
    def __init__(self, max_frames=100):
        super().__init__()
        self._processed_frames = set()
        self._frame_history = deque(maxlen=max_frames)
        self._start_time = None

        self._latency_history = []

    def _set_start_time(self, time: float):
        self._start_time = time

    def _reset_start_time(self):
        self._start_time = None

    async def on_push_frame(self, data: FramePushed):
        """Handle the event when a frame is pushed from one processor to another.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        transferred through the pipeline.

        Args:
            data (FramePushed): The event data containing details about the frame transfer.
        """
        self._processed_frames.add(data.frame.id)
        self._frame_history.append(data.frame.id)

        # If we've exceeded our history size, remove the oldest frame ID
        # from the set of processed frames.
        if len(self._processed_frames) > len(self._frame_history):
            # Rebuild the set from the current deque contents
            self._processed_frames = set(self._frame_history)

        if isinstance(data.frame, LLMFullResponseStartFrame):
            self._set_start_time(time.time())

        if isinstance(data.frame, TTSTextFrame):
            if self._start_time:
                response_time = time.time() - self._start_time
                logger.info(f"Agent response time: {response_time} seconds")
                self._latency_history.append(response_time)
                self._reset_start_time()

    def get_latency_history(self):
        return self._latency_history

    def get_average_latency(self):
        return sum(self._latency_history) / len(self._latency_history)

    def get_median_latency(self):
        return sorted(self._latency_history)[len(self._latency_history) // 2]

    def get_min_latency(self):
        return min(self._latency_history)

    def get_max_latency(self):
        return max(self._latency_history)

    def get_latency_stats(self):
        if not self._latency_history:
            return {
                "average": 0,
                "median": 0,
                "min": 0,
                "max": 0,
            }
        return {
            "average": self.get_average_latency(),
            "median": self.get_median_latency(),
            "min": self.get_min_latency(),
            "max": self.get_max_latency(),
        }
