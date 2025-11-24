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
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AggregationType(str, Enum):
    """Built-in aggregation strings."""

    SENTENCE = "sentence"
    WORD = "word"

    def __str__(self):
        return self.value


@dataclass
class Aggregation:
    """Data class representing aggregated text and its type.

    An Aggregation object is created whenever a stream of text is aggregated by
    a text aggregator. It contains the aggregated text and a type indicating
    the nature of the aggregation.

    Parameters:
        text: The aggregated text content.
        type: The type of aggregation the text represents (e.g., 'sentence', 'word', 'token',
              'my_custom_aggregation').
    """

    text: str
    type: str

    def __str__(self) -> str:
        """Return a string representation of the aggregation.

        Returns:
            A descriptive string showing the type and text of the aggregation.
        """
        return f"Aggregation by {self.type}: {self.text}"


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
    def text(self) -> Aggregation:
        """Get the currently aggregated text.

        Subclasses must implement this property to return the text that has
        been accumulated so far in their internal buffer or storage.

        Returns:
            The text that has been accumulated so far.
        """
        pass

    @abstractmethod
    async def aggregate(self, text: str) -> Optional[Aggregation]:
        """Aggregate the specified text with the currently accumulated text.

        This method should be implemented to define how the new text contributes
        to the aggregation process. It returns the aggregated text and a string
        describing how it was aggregated if it's ready to be processed,
        or None otherwise.

        Subclasses should implement their specific logic for:

        - How to combine new text with existing accumulated text
        - When to consider the aggregated text ready for processing
        - What criteria determine text completion (e.g., sentence boundaries)
        - When a completion occurs, the method should return an Aggregation object
          containing the aggregated text and its type. The text should be stripped
          of leading/trailing whitespace so that consumers can rely on a consistent
          format.

        Args:
            text: The text to be aggregated.

        Returns:
            An Aggregation object if ready for processing, or None if more
            text is needed before the aggregated content is ready. If an Aggregation
            object is returned, it should consist of the updated aggregated text,
            stripped of leading/trailing whitespace, and a string indicating the
            type of aggregation (e.g., 'sentence', 'word', 'token', 'my_custom_aggregation').
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
