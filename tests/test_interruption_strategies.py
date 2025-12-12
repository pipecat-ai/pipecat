#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.audio.interruptions.min_words_interruption_strategy import (
    MinWordsInterruptionStrategy as OldMinWordsInterruptionStrategy,
)
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.turns.vad_interruption_strategy import VADInterruptionStrategy


class TestOldMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_min_words(self):
        strategy = OldMinWordsInterruptionStrategy(min_words=2)
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


class TestMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_only_transcriptions(self):
        strategy = MinWordsInterruptionStrategy(min_words=2)

        should_interrupt = None

        @strategy.event_handler("on_should_interrupt")
        async def on_should_interrupt(strategy):
            nonlocal should_interrupt
            should_interrupt = True

        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertFalse(should_interrupt)

        await strategy.process_frame(
            TranscriptionFrame(text=" there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_interrupt)

        # Reset and check again
        should_interrupt = None
        await strategy.reset()

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertFalse(should_interrupt)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertTrue(should_interrupt)

    async def test_only_interim_transcriptions(self):
        strategy = MinWordsInterruptionStrategy(min_words=2)

        should_interrupt = None

        @strategy.event_handler("on_should_interrupt")
        async def on_should_interrupt(strategy):
            nonlocal should_interrupt
            should_interrupt = True

        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_interrupt)

        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_interrupt)

    async def test_all_transcriptions(self):
        strategy = MinWordsInterruptionStrategy(min_words=2)

        should_interrupt = None

        @strategy.event_handler("on_should_interrupt")
        async def on_should_interrupt(strategy):
            nonlocal should_interrupt
            should_interrupt = True

        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_interrupt)

        await strategy.process_frame(
            TranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_interrupt)


class TestVADInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_vad_strategy(self):
        strategy = VADInterruptionStrategy()

        should_interrupt = None

        @strategy.event_handler("on_should_interrupt")
        async def on_should_interrupt(strategy):
            nonlocal should_interrupt
            should_interrupt = True

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertFalse(should_interrupt)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_interrupt)
