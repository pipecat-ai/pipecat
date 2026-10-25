#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base turn analyzer for determining end-of-turn in audio conversations.

This module provides the abstract base class and enumeration for analyzing
when a user has finished speaking in a conversation.
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from pipecat.metrics.metrics import MetricsData


class EndOfTurnState(Enum):
    """State enumeration for end-of-turn analysis results.

    Parameters:
        COMPLETE: The user has finished their turn and stopped speaking.
        INCOMPLETE: The user is still speaking or may continue speaking.
    """

    COMPLETE = 1
    INCOMPLETE = 2


class BaseTurnParams(BaseModel):
    """Base class for turn analyzer parameters."""

    pass


class BaseTurnAnalyzer(ABC):
    """Abstract base class for analyzing user end of turn.

    This class inherits from BaseObject to leverage its event handling system
    while still defining an abstract interface through abstract methods.
    """

    def __init__(self, *, sample_rate: int | None = None):
        """Initialize the turn analyzer.

        Args:
            sample_rate: Optional initial sample rate for audio processing.
                If provided, this will be used as the fixed sample rate.
        """
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

    @property
    @abstractmethod
    def params(self) -> BaseTurnParams:
        """Get the current turn analyzer parameters.

        Returns:
            Current turn analyzer configuration parameters.
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
    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Analyzes if an end of turn has occurred based on the audio input.

        Returns:
            EndOfTurnState: The result of the end of turn analysis.
        """
        pass

    def pop_eager(self) -> bool:
        """Report an eager end of turn reached since the last call, if any.

        An eager end of turn is a verdict the analyzer is not yet ready to
        commit to: the evidence is good enough to start generating a response,
        but not yet good enough to end the turn. A strategy that polls this
        after :meth:`append_audio` and :meth:`analyze_end_of_turn` (see
        :class:`~pipecat.turns.user_stop.TurnAnalyzerEagerUserTurnStopStrategy`)
        answers it speculatively, so the response is ready when the turn
        commits and is discarded if the user resumes speaking instead.

        Analyzers that reach eager ends of turn override this to return True
        once per eager end of turn and False until the next one. The committed
        verdict still arrives through :meth:`append_audio` or
        :meth:`analyze_end_of_turn` as before. The default never reaches one.

        Returns:
            bool: True if an eager end of turn was reached since the last call.
        """
        return False

    def update_vad_start_secs(self, vad_start_secs: float):
        """Update the VAD start trigger time.

        The turn analyzer may choose to change its buffer size depending
        on this value.

        Args:
            vad_start_secs (float): The number of seconds of voice activity
                before triggering the user speaking event.
        """
        pass

    @abstractmethod
    def clear(self):
        """Reset the turn analyzer to its initial state."""
        pass

    async def cleanup(self):
        """Cleanup the turn analyzer."""
        pass
