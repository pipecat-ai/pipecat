#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseTextFilter(ABC):
    @abstractmethod
    def update_settings(self, settings: Mapping[str, Any]):
        pass

    @abstractmethod
    def filter(self, text: str) -> str:
        pass
