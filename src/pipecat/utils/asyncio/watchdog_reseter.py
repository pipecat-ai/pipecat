#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class WatchdogReseter(ABC):
    @abstractmethod
    def reset_watchdog(self):
        pass
