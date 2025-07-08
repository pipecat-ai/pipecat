#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base audio mixer for output transport integration.

Provides the abstract base class for audio mixers that can be integrated with
output transports to mix incoming audio with generated audio from the mixer.
"""

from abc import ABC, abstractmethod

from pipecat.frames.frames import MixerControlFrame


class BaseAudioMixer(ABC):
    """Base class for output transport audio mixers.

    This is a base class for output transport audio mixers. If an audio mixer
    is provided to the output transport it will be used to mix the audio frames
    coming into to the transport with the audio generated from the mixer. There
    are control frames to update mixer settings or to enable or disable the
    mixer at runtime.
    """

    @abstractmethod
    async def start(self, sample_rate: int):
        """Initialize the mixer when the output transport starts.

        This will be called from the output transport when the transport is
        started. It can be used to initialize the mixer. The output transport
        sample rate is provided so the mixer can adjust to that sample rate.

        Args:
            sample_rate: The sample rate of the output transport in Hz.
        """
        pass

    @abstractmethod
    async def stop(self):
        """Clean up the mixer when the output transport stops.

        This will be called from the output transport when the transport is
        stopping.
        """
        pass

    @abstractmethod
    async def process_frame(self, frame: MixerControlFrame):
        """Process mixer control frames from the transport.

        This will be called when the output transport receives a
        MixerControlFrame.

        Args:
            frame: The mixer control frame to process.
        """
        pass

    @abstractmethod
    async def mix(self, audio: bytes) -> bytes:
        """Mix transport audio with mixer-generated audio.

        This is called with the audio that is about to be sent from the
        output transport and that should be mixed with the mixer audio if the
        mixer is enabled.

        Args:
            audio: Raw audio bytes from the transport to mix.

        Returns:
            Mixed audio bytes combining transport and mixer audio.
        """
        pass
