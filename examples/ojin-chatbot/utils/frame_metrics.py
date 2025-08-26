# Copyright (c) 2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    OutputImageRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FrameMetricsProcessor(FrameProcessor):
    """Measures video output FPS and time-to-first-frame (TTFF) after user stops speaking.

    - FPS is computed over a sliding time window and logged periodically.
    - TTFF is measured from `UserStoppedSpeakingFrame` to the next `OutputImageRawFrame`.

    Args:
        log_interval_sec: Interval in seconds to log FPS statistics.
        name: Optional processor name.
    """

    def __init__(self, *, log_interval_sec: float = 2.0, name: Optional[str] = None):
        super().__init__(name=name)
        self._log_interval_ns: int = int(log_interval_sec * 1e9)
        self._last_fps_report_time_ns: Optional[int] = None
        self._frame_count_since_report: int = 0

        # TTFF state
        self._ttff_start_ns: Optional[int] = None
        # Instant FPS state
        self._last_frame_time_ns: Optional[int] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Let base handle Start/Cancel/Pause/Resume/etc first
        await super().process_frame(frame, direction)

        now_ns = self._now_ns()

        # Capture TTFF start when user stops speaking
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._ttff_start_ns = now_ns
            # Continue forwarding the system frame
            await self.push_frame(frame, direction)
            return

        # Handle output image frames for FPS and TTFF end
        if isinstance(frame, OutputImageRawFrame):
            # FPS accounting
            if self._last_fps_report_time_ns is None:
                self._last_fps_report_time_ns = now_ns
                self._frame_count_since_report = 0
            self._frame_count_since_report += 1

            elapsed_ns = now_ns - self._last_fps_report_time_ns
            if elapsed_ns >= self._log_interval_ns and elapsed_ns > 0:
                fps = self._frame_count_since_report / (elapsed_ns / 1e9)
                logger.info(f"[FrameMetrics] Output FPS: {fps:.2f} over {elapsed_ns/1e9:.2f}s")
                # Reset window
                self._last_fps_report_time_ns = now_ns
                self._frame_count_since_report = 0

            # Instant FPS for UI graph
            if self._last_frame_time_ns is not None and now_ns > self._last_frame_time_ns:
                interval_s = (now_ns - self._last_frame_time_ns) / 1e9
                if interval_s > 0:
                    instant_fps = 1.0 / interval_s
                    _fps_history.append(instant_fps)
                    _current_fps_state.value = instant_fps
            self._last_frame_time_ns = now_ns

            # TTFF measure (if requested by last user stop)
            if self._ttff_start_ns is not None:
                ttff_ms = (now_ns - self._ttff_start_ns) / 1e6
                if ttff_ms >= 0 and not math.isnan(ttff_ms) and not math.isinf(ttff_ms):
                    logger.info(f"[FrameMetrics] TTFF after user stop: {ttff_ms:.1f} ms")
                self._ttff_start_ns = None

            await self.push_frame(frame, direction)
            return

        # Forward any other frame unchanged
        await self.push_frame(frame, direction)

    def _now_ns(self) -> int:
        try:
            return self.get_clock().get_time()
        except Exception:
            # Fallback to zero if clock not initialized yet; metrics will log later
            return 0


# ---------------------------
# Module-level simple state for UI
# ---------------------------

_fps_history: Deque[float] = deque(maxlen=240)  # ~4s at 60fps


class _CurrentFPS:
    def __init__(self):
        self.value: float = 0.0


_current_fps_state = _CurrentFPS()


def get_fps_history() -> List[float]:
    """Return a copy of recent FPS samples for rendering a graph."""
    return list(_fps_history)


def get_current_fps(window: int = 25) -> float:
    """Return the average FPS over the last `window` samples (default 25).

    Falls back to the latest instant FPS if there is no history yet.
    """
    try:
        w = int(window)
    except Exception:
        w = 25
    if w <= 0:
        w = 1

    if not _fps_history:
        return float(_current_fps_state.value)

    # Average last w samples (or fewer if history is short)
    samples = list(_fps_history)
    n = min(w, len(samples))
    if n == 0:
        return 0.0
    total = 0.0
    for v in samples[-n:]:
        total += float(v)
    return total / n
