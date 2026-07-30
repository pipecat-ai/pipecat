#
# Copyright (c) 2024-2026, Daily
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

    @property
    def is_passthrough(self) -> bool:
        """Whether this mixer contributes nothing to the outgoing audio.

        Configuring a mixer puts the output transport on its continuous send
        path: the media sender synthesizes and mixes a silence frame whenever
        its queue is empty, so every leg runs a full-rate mix/serialize/write
        loop even while nobody is speaking, and an interruption drains the audio
        queue in place rather than cancelling and recreating the audio task.
        That is correct for a mixer that actually generates audio (background
        noise, hold music) — cancelling the task there would leave an audible
        gap in the background bed.

        A mixer that only ever returns its input unchanged — e.g. a silence
        mixer installed unconditionally when ambient audio is disabled — pays
        that cost for nothing. Returning True lets the transport treat it as if
        no mixer were configured.

        Returns:
            False by default, so existing mixers are unaffected.
        """
        return False

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
