#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base text aggregator interface for Pipecat text processing.

This module defines the abstract base class for text aggregators that accumulate
and process text tokens, typically used by TTS services to determine when
aggregated text should be sent for speech synthesis.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseTextAggregator(ABC):
    """Base class for text aggregators in the Pipecat framework.

    Text aggregators are usually used by the TTS service to aggregate LLM tokens
    and decide when the aggregated text should be pushed to the TTS service.

    Text aggregators can also be used to manipulate text while it's being
    aggregated (e.g. reasoning blocks can be removed).

    Subclasses must implement all abstract methods to define specific aggregation
    logic, text manipulation behavior, and state management for interruptions.
    """

    @property
    @abstractmethod
    def text(self) -> str:
        """Get the currently aggregated text.

        Subclasses must implement this property to return the text that has
        been accumulated so far in their internal buffer or storage.

        Returns:
            The text that has been accumulated so far.
        """
        pass

    @abstractmethod
    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregate the specified text with the currently accumulated text.

        This method should be implemented to define how the new text contributes
        to the aggregation process. It returns the updated aggregated text if
        it's ready to be processed, or None otherwise.

        Subclasses should implement their specific logic for:

        - How to combine new text with existing accumulated text
        - When to consider the aggregated text ready for processing
        - What criteria determine text completion (e.g., sentence boundaries)

        Args:
            text: The text to be aggregated.

        Returns:
            The updated aggregated text if ready for processing, or None if more
            text is needed before the aggregated content is ready.
        """
        pass

    @abstractmethod
    async def handle_interruption(self):
        """Handle interruptions in the text aggregation process.

        When an interruption occurs it is possible that we might want to discard
        the aggregated text or do some internal modifications to the aggregated text.

        Subclasses should implement this method to define how they respond to
        interruptions, such as clearing buffers, resetting state, or preserving
        partial content.
        """
        pass

    @abstractmethod
    async def reset(self):
        """Clear the internally aggregated text and reset to initial state.

        Subclasses should implement this method to return the aggregator to its
        initial state, discarding any previously accumulated text content and
        resetting any internal tracking variables.
        """
        pass
