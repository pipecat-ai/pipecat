#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseTextFilter(ABC):
    @abstractmethod
    async def update_settings(self, settings: Mapping[str, Any]):
        pass

    @abstractmethod
    async def filter(self, text: str) -> str:
        pass

    @abstractmethod
    async def handle_interruption(self):
        pass

    @abstractmethod
    async def reset_interruption(self):
        pass
