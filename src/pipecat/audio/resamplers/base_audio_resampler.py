#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base audio resampler interface for Pipecat.

This module defines the abstract base class for audio resampling implementations,
providing a common interface for converting audio between different sample rates.
"""

from abc import ABC, abstractmethod
from typing import Literal

SoxrQuality = Literal["QQ", "LQ", "MQ", "HQ", "VHQ"]
"""SOXR resampling quality presets, from fastest (QQ) to highest quality (VHQ)."""


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

    async def flush(self) -> bytes:
        """Emit any audio still held internally and reset the resampler.

        Streaming resamplers keep recently seen input in their filter, so the
        tail of a stream is only emitted once more audio arrives. Call this at
        the end of a continuous stream to collect that tail and start clean.

        Returns:
            The remaining resampled audio as raw bytes, empty if the resampler
            holds nothing.
        """
        return b""

    async def reset(self):
        """Discard any audio still held internally and reset the resampler.

        Use this when a stream is abandoned rather than finished (e.g. after an
        interruption), so its leftover tail doesn't leak into the next stream.
        """
        pass
