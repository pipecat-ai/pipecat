#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class BaseAudioResampler(ABC):
    """Abstract base class for audio resampling. This class defines an
    interface for audio resampling implementations.
    """

    @abstractmethod
    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """
        Resamples the given audio data to a different sample rate.

        This is an abstract method that must be implemented in subclasses.

        Parameters:
            audio (bytes): The audio data to be resampled, represented as a byte string.
            in_rate (int): The original sample rate of the audio data (in Hz).
            out_rate (int): The desired sample rate for the resampled audio data (in Hz).

        Returns:
            bytes: The resampled audio data as a byte string.
        """
        pass
