#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)


class TestMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_bot_speaking_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text=" there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

        # Reset and check again
        should_start = None
        await strategy.reset()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_start)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_speaking_all_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_not_speaking_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertTrue(should_start)

    async def test_bot_not_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)


class TestVADUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_vad_strategy(self):
        strategy = VADUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)


class TestTranscriptionUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_transcription_strategy(self):
        strategy = TranscriptionUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        self.assertFalse(should_start)

        await strategy.process_frame(BotStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        self.assertTrue(should_start)
