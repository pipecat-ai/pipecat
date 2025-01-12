#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod

from pipecat.frames.frames import FilterControlFrame


class BaseAudioFilter(ABC):
    """This is a base class for input transport audio filters. If an audio
    filter is provided to the input transport it will be used to process audio
    before VAD and before pushing it downstream. There are control frames to
    update filter settings or to enable or disable the filter at runtime.

    """

    @abstractmethod
    async def start(self, sample_rate: int):
        """This will be called from the input transport when the transport is
        started. It can be used to initialize the filter. The input transport
        sample rate is provided so the filter can adjust to that sample rate.

        """
        pass

    @abstractmethod
    async def stop(self):
        """This will be called from the input transport when the transport is
        stopping.

        """
        pass

    @abstractmethod
    async def process_frame(self, frame: FilterControlFrame):
        """This will be called when the input transport receives a
        FilterControlFrame.

        """
        pass

    @abstractmethod
    async def filter(self, audio: bytes) -> bytes:
        pass
