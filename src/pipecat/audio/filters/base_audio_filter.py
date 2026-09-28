#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base audio filter interface for transport audio processing.

This module provides the abstract base class for implementing audio filters
that process audio data before VAD and downstream processing in input transports,
or before the mixer and the write in output transports.
"""

from abc import ABC, abstractmethod

from pipecat.frames.frames import FilterControlFrame


class BaseAudioFilter(ABC):
    """Base class for transport audio filters.

    This is a base class for transport audio filters. If an audio filter is
    provided to the input transport it will be used to process audio before VAD
    and before pushing it downstream. If an audio filter is provided to the
    output transport it will be used to process the audio it receives before
    mixing and writing it. There are control frames to update filter settings
    or to enable or disable the filter at runtime.
    """

    @abstractmethod
    async def start(self, sample_rate: int):
        """Initialize the filter when the transport is set up.

        Called once, from the transport's setup, and paired with :meth:`stop`.
        The transport sample rate is provided so the filter can adjust to that
        sample rate.

        Args:
            sample_rate: The sample rate of the transport in Hz.
        """
        pass

    @abstractmethod
    async def stop(self):
        """Clean up the filter when the transport is cleaned up.

        Called once, from the transport's cleanup, and paired with
        :meth:`start`.
        """
        pass

    @abstractmethod
    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames for runtime filter configuration.

        This will be called when the transport receives a FilterControlFrame.

        Args:
            frame: The control frame containing filter commands or settings.
        """
        pass

    @abstractmethod
    async def filter(self, audio: bytes) -> bytes:
        """Apply the audio filter to the provided audio data.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Filtered audio data as bytes.
        """
        pass
