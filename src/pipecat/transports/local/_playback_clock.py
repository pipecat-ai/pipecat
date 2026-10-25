#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Played speech duration from PortAudio callback DAC times and its stream clock."""

import math
from collections import defaultdict


class PlaybackClock:
    """Track valid speech spans separately from device padding and background audio.

    Access is protected by the output transport's callback lock. A callback time
    that is absent or invalid disables measurement for that output.
    """

    def __init__(self):
        """Create empty per-output device schedules."""
        self.spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.unknown: set[str] = set()

    def schedule(self, context_id: str, start: float, duration: float, *, valid: bool):
        """Record a speech span in the PortAudio stream clock domain."""
        if not valid or not math.isfinite(start) or start <= 0:
            self.unknown.add(context_id)
        spans = self.spans[context_id]
        if len(spans) >= 65536:
            raise RuntimeError("Device playback schedule limit exceeded")
        spans.append((start, duration))

    def end(self, context_id: str) -> float:
        """Last scheduled device time for the output, or zero for no speech."""
        return max(
            (start + duration for start, duration in self.spans.get(context_id, [])), default=0
        )

    def played_ms(self, context_id: str, now: float) -> float | None:
        """Count only speech samples whose DAC time has passed."""
        if context_id in self.unknown or not math.isfinite(now):
            return None
        return (
            sum(
                min(duration, max(0.0, now - start))
                for start, duration in self.spans.get(context_id, [])
            )
            * 1000
        )

    def forget(self, context_id: str):
        """Release a schedule after its terminal report."""
        self.spans.pop(context_id, None)
        self.unknown.discard(context_id)
