#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base audio resampler interface for Pipecat.

This module defines the abstract base class for audio resampling implementations,
providing a common interface for converting audio between different sample rates.
"""

from abc import ABC, abstractmethod


class BaseAudioResampler(ABC):
    """Abstract base class for audio resampling implementations.

    This class defines the interface that all audio resampling implementations
    must follow, providing a standardized way to convert audio data between
    different sample rates.
    """

    @abstractmethod
    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resamples the given audio data to a different sample rate.

        This is an abstract method that must be implemented in subclasses.

        Args:
            audio: The audio data to be resampled, as raw bytes.
            in_rate: The original sample rate of the audio data in Hz.
            out_rate: The desired sample rate for the output audio in Hz.

        Returns:
            The resampled audio data as raw bytes.
        """
        pass
