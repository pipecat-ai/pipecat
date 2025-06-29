#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base clock interface for Pipecat timing operations."""

from abc import ABC, abstractmethod


class BaseClock(ABC):
    """Abstract base class for clock implementations.

    Provides a common interface for timing operations used in Pipecat
    for synchronization, scheduling, and time-based processing.
    """

    @abstractmethod
    def get_time(self) -> int:
        """Get the current time value.

        Returns:
            The current time as an integer value. The specific unit and
            reference point depend on the concrete implementation.
        """
        pass

    @abstractmethod
    def start(self):
        """Start or initialize the clock.

        Performs any necessary initialization or starts the timing mechanism.
        This method should be called before using get_time().
        """
        pass
