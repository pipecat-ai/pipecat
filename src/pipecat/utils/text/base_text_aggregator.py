#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import Optional


class BaseTextAggregator(ABC):
    """This is the base class for text aggregators. Text aggregators are usually
    used by the TTS service to aggregate LLM tokens and decide when the
    aggregated text should be pushed to the TTS service.

    Text aggregators can also be used to manipulate text while it's being
    aggregated (e.g. reasoning blocks can be removed).

    """

    @property
    @abstractmethod
    def text(self) -> str:
        """Returns the currently aggregated text."""
        pass

    @abstractmethod
    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregates the specified text with the currently accumulated text.

        This method should be implemented to define how the new text contributes
        to the aggregation process. It returns the updated aggregated text if
        it's ready to be processed, or None otherwise.

        Args:
            text (str): The text to be aggregated.

        Returns:
            Optional[str]: The updated aggregated text or None if aggregated
            text is not ready.

        """
        pass

    @abstractmethod
    async def handle_interruption(self):
        """Handles interruptions. When an interruption occurs it is possible
        that we might want to discard the aggregated text or do some internal
        modifications to the aggregated text.

        """
        pass

    @abstractmethod
    async def reset(self):
        """Clears the internally aggregated text."""
        pass
