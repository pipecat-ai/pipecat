#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

from pipecat.metrics.metrics import MetricsData


class EndOfTurnState(Enum):
    COMPLETE = 1
    INCOMPLETE = 2


class BaseTurnAnalyzer(ABC):
    """Abstract base class for analyzing user end of turn.

    This class inherits from BaseObject to leverage its event handling system
    while still defining an abstract interface through abstract methods.
    """

    def __init__(self, *, sample_rate: Optional[int] = None):
        self._init_sample_rate = sample_rate
        self._sample_rate = 0

    @property
    def sample_rate(self) -> int:
        """Returns the current sample rate.

        Returns:
            int: The effective sample rate for audio processing.
        """
        return self._sample_rate

    def set_sample_rate(self, sample_rate: int):
        """Sets the sample rate for audio processing.

        If the initial sample rate was provided, it will use that; otherwise, it sets to
        the provided sample rate.

        Args:
            sample_rate (int): The sample rate to set.
        """
        self._sample_rate = self._init_sample_rate or sample_rate

    @property
    @abstractmethod
    def speech_triggered(self) -> bool:
        """Determines if speech has been detected.

        Returns:
            bool: True if speech is triggered, otherwise False.
        """
        pass

    @abstractmethod
    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Appends audio data for analysis.

        Args:
            buffer (bytes): The audio data to append.
            is_speech (bool): Indicates whether the appended audio is speech or not.

        Returns:
            EndOfTurnState: The resulting state after appending the audio.
        """
        pass

    @abstractmethod
    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        """Analyzes if an end of turn has occurred based on the audio input.

        Returns:
            EndOfTurnState: The result of the end of turn analysis.
        """
        pass

    @abstractmethod
    def clear(self):
        """Reset the turn analyzer to its initial state."""
        pass
