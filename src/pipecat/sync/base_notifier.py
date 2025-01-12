#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    @abstractmethod
    async def notify(self):
        pass

    @abstractmethod
    async def wait(self):
        pass
