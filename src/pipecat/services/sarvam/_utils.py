#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility helpers for Sarvam services."""

import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    """Collect values and flush them periodically.

    Args:
        callback: Function called with the accumulated value on flush.
        duration: Minimum elapsed wall-clock duration before automatic flush.
    """

    def __init__(
        self,
        callback: Callable[[T], None],
        *,
        duration: float,
    ) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value and flush if the interval elapsed."""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]

        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Flush any accumulated value."""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()
