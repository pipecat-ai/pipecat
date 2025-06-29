#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base text filter interface for Pipecat text processing.

This module defines the abstract base class for text filters that can modify
text content in the processing pipeline, including support for settings updates
and interruption handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseTextFilter(ABC):
    """Abstract base class for text filters in the Pipecat framework.

    Text filters are responsible for modifying text content as it flows through
    the processing pipeline. They support dynamic settings updates and can handle
    interruptions to reset their internal state.

    Subclasses must implement all abstract methods to define specific filtering
    behavior, settings management, and interruption handling logic.
    """

    @abstractmethod
    async def update_settings(self, settings: Mapping[str, Any]):
        """Update the filter's configuration settings.

        Subclasses should implement this method to handle dynamic configuration
        updates during runtime, updating internal state as needed.

        Args:
            settings: Dictionary of setting names to values for configuration.
        """
        pass

    @abstractmethod
    async def filter(self, text: str) -> str:
        """Apply filtering transformations to the input text.

        Subclasses must implement this method to define the specific text
        transformations that should be applied to the input.

        Args:
            text: The input text to be filtered.

        Returns:
            The filtered text after applying transformations.
        """
        pass

    @abstractmethod
    async def handle_interruption(self):
        """Handle interruption events in the processing pipeline.

        Subclasses should implement this method to reset internal state,
        clear buffers, or perform other cleanup when an interruption occurs.
        """
        pass

    @abstractmethod
    async def reset_interruption(self):
        """Reset the filter state after an interruption has been handled.

        Subclasses should implement this method to restore the filter to normal
        operation after an interruption has been processed and resolved.
        """
        pass
