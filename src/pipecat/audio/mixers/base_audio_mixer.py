#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod

from pipecat.frames.frames import MixerControlFrame


class BaseAudioMixer(ABC):
    """This is a base class for output transport audio mixers. If an audio mixer
    is provided to the output transport it will be used to mix the audio frames
    coming into to the transport with the audio generated from the mixer. There
    are control frames to update mixer settings or to enable or disable the
    mixer at runtime.

    """

    @abstractmethod
    async def start(self, sample_rate: int):
        """This will be called from the output transport when the transport is
        started. It can be used to initialize the mixer. The output transport
        sample rate is provided so the mixer can adjust to that sample rate.

        """
        pass

    @abstractmethod
    async def stop(self):
        """This will be called from the output transport when the transport is
        stopping.

        """
        pass

    @abstractmethod
    async def process_frame(self, frame: MixerControlFrame):
        """This will be called when the output transport receives a
        MixerControlFrame.

        """
        pass

    @abstractmethod
    async def mix(self, audio: bytes) -> bytes:
        """This is called with the audio that is about to be sent from the
        output transport and that should be mixed with the mixer audio if the
        mixer is enabled.

        """
        pass
