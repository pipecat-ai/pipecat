#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class AudioFilter(ABC):
    @abstractmethod
    async def filter(self, audio: bytes, sample_rate: int, num_channels: int) -> bytes:
        pass
