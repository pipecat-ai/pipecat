#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User response aggregation for text frames.

This module provides an aggregator that collects user responses and outputs
them as TextFrame objects, useful for capturing and processing user input
in conversational pipelines.
"""

from pipecat.frames.frames import TextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.utils.string import concatenate_aggregated_text


class UserResponseAggregator(LLMUserAggregator):
    """Aggregates user responses into TextFrame objects.

    This aggregator extends LLMUserAggregator to specifically handle
    user input by collecting text responses and outputting them as TextFrame
    objects when the aggregation is complete.
    """

    def __init__(self, **kwargs):
        """Initialize the user response aggregator.

        .. deprecated:: 0.0.92
            `UserResponseAggregator` is deprecated and will be removed in a future version.

        Args:
            **kwargs: Additional arguments passed to parent LLMUserAggregator.
        """
        super().__init__(context=LLMContext(), **kwargs)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`UserResponseAggregator` is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

    async def push_aggregation(self) -> str:
        """Push the aggregated user response as a TextFrame.

        Creates a TextFrame from the current aggregation if it contains content,
        resets the aggregation state, and pushes the frame downstream.

        Returns:
            The pushed aggregation text, or empty string if nothing to push.
        """
        if len(self._aggregation) > 0:
            text = concatenate_aggregated_text(self._aggregation).strip()
            frame = TextFrame(text)

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = []

            await self.push_frame(frame)

            # Reset our accumulator state.
            await self.reset()

            return text
        return ""
