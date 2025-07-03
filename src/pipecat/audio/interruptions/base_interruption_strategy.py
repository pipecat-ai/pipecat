#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base interruption strategy for determining when users can interrupt bot speech."""

from abc import ABC, abstractmethod


class BaseInterruptionStrategy(ABC):
    """Base class for interruption strategies.

    This is a base class for interruption strategies. Interruption strategies
    decide when the user can interrupt the bot while the bot is speaking. For
    example, there could be strategies based on audio volume or strategies based
    on the number of words the user spoke.
    """

    async def append_audio(self, audio: bytes, sample_rate: int):
        """Append audio data to the strategy for analysis.

        Not all strategies handle audio. Default implementation does nothing.

        Args:
            audio: Raw audio bytes to append.
            sample_rate: Sample rate of the audio data in Hz.
        """
        pass

    async def append_text(self, text: str):
        """Append text data to the strategy for analysis.

        Not all strategies handle text. Default implementation does nothing.

        Args:
            text: Text string to append for analysis.
        """
        pass

    @abstractmethod
    async def should_interrupt(self) -> bool:
        """Determine if the user should interrupt the bot.

        This is called when the user stops speaking and it's time to decide
        whether the user should interrupt the bot. The decision will be based on
        the aggregated audio and/or text.

        Returns:
            True if the user should interrupt the bot, False otherwise.
        """
        pass

    @abstractmethod
    async def reset(self):
        """Reset the current accumulated text and/or audio."""
        pass
