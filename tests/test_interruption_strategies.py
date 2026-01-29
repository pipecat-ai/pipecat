#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy


class TestMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_min_words(self):
        strategy = MinWordsInterruptionStrategy(min_words=2)
        await strategy.append_text("Hello")
        self.assertEqual(await strategy.should_interrupt(), False)
        await strategy.append_text(" there!")
        self.assertEqual(await strategy.should_interrupt(), True)
        # Reset and check again
        await strategy.reset()
        await strategy.append_text("Hello!")
        self.assertEqual(await strategy.should_interrupt(), False)
        await strategy.append_text(" How are you?")
        self.assertEqual(await strategy.should_interrupt(), True)


if __name__ == "__main__":
    unittest.main()
