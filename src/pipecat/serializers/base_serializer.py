#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod

from pipecat.frames.frames import Frame


class FrameSerializer(ABC):

    @abstractmethod
    def serialize(self, frame: Frame) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Frame | None:
        pass
