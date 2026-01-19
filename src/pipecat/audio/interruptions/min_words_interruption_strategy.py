#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimum words interruption strategy for word count-based interruptions."""

from loguru import logger

from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy


class MinWordsInterruptionStrategy(BaseInterruptionStrategy):
    """Interruption strategy based on minimum number of words spoken.

    This is an interruption strategy based on a minimum number of words said
    by the user. That is, the strategy will be true if the user has said at
    least that amount of words.

    .. deprecated:: 0.0.99

        This class is deprecated, use
        `pipecat.turns.user_start.MinWordsUserTurnStartStrategy` with `PipelineTask`'s
        new `user_turn_strategies` parameter instead.

    """

    def __init__(self, *, min_words: int):
        """Initialize the minimum words interruption strategy.

        Args:
            min_words: Minimum number of words required to trigger an interruption.
        """
        super().__init__()
        self._min_words = min_words
        self._text = ""

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'pipecat.audio.interruptions' is deprecated. "
                "Use `pipecat.turns.user_start.MinWordsUserTurnStartStrategy` with `PipelineTask`'s "
                "new `user_turn_strategies` parameter instead.",
                DeprecationWarning,
            )

    async def append_text(self, text: str):
        """Append text for word count analysis.

        Args:
            text: Text string to append to the accumulated text.

        Note: Not all strategies need to handle text.
        """
        self._text += text

    async def should_interrupt(self) -> bool:
        """Check if the minimum word count has been reached.

        Returns:
            True if the user has spoken at least the minimum number of words.
        """
        word_count = len(self._text.split())
        interrupt = word_count >= self._min_words
        logger.debug(
            f"should_interrupt={interrupt} num_spoken_words={word_count} min_words={self._min_words}"
        )
        return interrupt

    async def reset(self):
        """Reset the accumulated text for the next analysis cycle."""
        self._text = ""
