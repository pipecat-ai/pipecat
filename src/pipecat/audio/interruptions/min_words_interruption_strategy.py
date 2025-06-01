#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy


class MinWordsInterruptionStrategy(BaseInterruptionStrategy):
    """This is an interruption strategy based on a minimum number of words said
    by the user. That is, the strategy will be true if the user has said at
    least that amount of words.

    """

    def __init__(self, *, min_words: int):
        super().__init__()
        self._min_words = min_words
        self._text = ""

    async def append_text(self, text: str):
        """Appends text for later analysis. Not all strategies need to handle
        text.

        """
        self._text += text

    async def should_interrupt(self) -> bool:
        word_count = len(self._text.split())
        interrupt = word_count >= self._min_words
        logger.debug(
            f"should_interrupt={interrupt} num_spoken_words={word_count} min_words={self._min_words}"
        )
        return interrupt

    async def reset(self):
        self._text = ""
