#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class BaseClock(ABC):
    @abstractmethod
    def get_time(self) -> int:
        pass

    @abstractmethod
    def start(self):
        pass
