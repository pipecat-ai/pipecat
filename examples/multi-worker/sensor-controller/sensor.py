#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Temperature sensor processors for the sensor-controller example.

Two custom :class:`FrameProcessor` subclasses that give the worker
pipeline real autonomous frame flow:

- :class:`SensorReader` simulates a thermometer. It runs an async tick
  loop that advances ``current`` toward ``target`` with a first-order
  lag plus Gaussian noise, and pushes a :class:`SensorReadingFrame` on
  every tick. ``target`` and ``response_rate`` are mutable so the
  worker's LLM can adjust them via tool calls.
- :class:`SensorStats` consumes the readings, maintains a rolling
  window, and exposes ``current`` / ``min`` / ``max`` / ``avg`` /
  ``trend`` as properties. The worker LLM reads these directly when
  answering the user.
"""

import random
import time
from collections import deque
from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class SensorReadingFrame(DataFrame):
    """A single temperature reading emitted by :class:`SensorReader`.

    Parameters:
        temperature: The reading in degrees Celsius.
        timestamp: Unix timestamp when the reading was taken.
    """

    temperature: float = 0.0
    timestamp: float = 0.0


class SensorReader(FrameProcessor):
    """Simulated temperature sensor with adjustable target and response rate.

    Each tick, ``current`` is updated as::

        current += (target - current) * response_rate + gauss(0, noise_sigma)

    This is a first-order lag toward ``target``. With ``response_rate=0.05``
    and a 1s tick, the current reading reaches ~halfway to target in ~14s;
    with ``response_rate=0.2`` it converges in ~5–10s.
    """

    def __init__(
        self,
        *,
        start_temp: float = 22.0,
        sample_period_s: float = 1.0,
        response_rate: float = 0.05,
        noise_sigma: float = 0.1,
    ):
        """Initialize the sensor.

        Args:
            start_temp: Initial temperature and initial target (°C).
            sample_period_s: Seconds between successive readings.
            response_rate: Fraction of the gap toward target closed each tick
                (clamped to ``[0.0, 1.0]``).
            noise_sigma: Standard deviation of the Gaussian noise added to
                each reading.
        """
        super().__init__()
        self._current = start_temp
        self._target = start_temp
        self._response_rate = max(0.0, min(1.0, response_rate))
        self._noise_sigma = noise_sigma
        self._sample_period_s = sample_period_s
        self._tick_task = None

    @property
    def current(self) -> float:
        """The most recent temperature reading (°C)."""
        return self._current

    @property
    def target(self) -> float:
        """The temperature the sensor is drifting toward (°C)."""
        return self._target

    @property
    def response_rate(self) -> float:
        """Fraction of the target-current gap closed per tick."""
        return self._response_rate

    def set_target(self, value: float) -> None:
        """Set a new target temperature (°C)."""
        self._target = value

    def set_response_rate(self, rate: float) -> None:
        """Set how aggressively the sensor approaches the target.

        Args:
            rate: Fraction in ``[0.0, 1.0]``. Clamped to that range.
        """
        self._response_rate = max(0.0, min(1.0, rate))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._tick_task is None:
            self._tick_task = self.create_task(self._tick_loop(), "ticker")
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        if self._tick_task is not None:
            await self.cancel_task(self._tick_task)
            self._tick_task = None
        await super().cleanup()

    async def _tick_loop(self) -> None:
        import asyncio

        while True:
            await asyncio.sleep(self._sample_period_s)
            gap = self._target - self._current
            self._current += gap * self._response_rate + random.gauss(0, self._noise_sigma)
            await self.push_frame(
                SensorReadingFrame(temperature=self._current, timestamp=time.time()),
                FrameDirection.DOWNSTREAM,
            )


class SensorStats(FrameProcessor):
    """Rolling-window statistics over :class:`SensorReadingFrame`s.

    Consumes readings as they flow downstream and exposes rolling
    ``min`` / ``max`` / ``avg`` / ``trend`` as properties — the worker
    LLM reads them directly when responding to the user.
    """

    def __init__(self, window: int = 30):
        """Initialize the stats aggregator.

        Args:
            window: Number of recent readings to retain.
        """
        super().__init__()
        self._readings: deque[float] = deque(maxlen=window)

    @property
    def current(self) -> float:
        """The most recent reading, or 0.0 if none have been seen."""
        return self._readings[-1] if self._readings else 0.0

    @property
    def min(self) -> float:
        return min(self._readings) if self._readings else 0.0

    @property
    def max(self) -> float:
        return max(self._readings) if self._readings else 0.0

    @property
    def avg(self) -> float:
        return sum(self._readings) / len(self._readings) if self._readings else 0.0

    @property
    def trend(self) -> str:
        """``"rising"`` / ``"falling"`` / ``"stable"`` based on first vs. last half of the window."""
        if len(self._readings) < 4:
            return "stable"
        half = len(self._readings) // 2
        old_avg = sum(list(self._readings)[:half]) / half
        new_avg = sum(list(self._readings)[half:]) / (len(self._readings) - half)
        diff = new_avg - old_avg
        if abs(diff) < 0.25:
            return "stable"
        return "rising" if diff > 0 else "falling"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, SensorReadingFrame):
            self._readings.append(frame.temperature)
        await self.push_frame(frame, direction)
