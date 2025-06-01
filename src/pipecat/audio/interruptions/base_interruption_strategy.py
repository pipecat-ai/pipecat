#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod


class BaseInterruptionStrategy(ABC):
    """This is a base class for interruption strategies. Interruption strategies
    decide when the user can interrupt the bot while the bot is speaking. For
    example, there could be strategies based on audio volume or strategies based
    on the number of words the user spoke.

    """

    async def append_audio(self, audio: bytes, sample_rate: int):
        """Appends audio to the strategy. Not all strategies handle audio."""
        pass

    async def append_text(self, text: str):
        """Appends text to the strategy. Not all strategies handle text."""
        pass

    @abstractmethod
    async def should_interrupt(self) -> bool:
        """This is called when the user stops speaking and it's time to decide
        whether the user should interrupt the bot. The decision will be based on
        the aggregated audio and/or text.

        """
        pass

    @abstractmethod
    async def reset(self):
        """Reset the current accumulated text and/or audio."""
        pass
