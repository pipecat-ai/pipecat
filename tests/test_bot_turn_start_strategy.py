#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.bot.timeout_bot_turn_start_strategy import TimeoutBotTurnStartStrategy
from pipecat.turns.bot.transcription_bot_turn_start_strategy import (
    TranscriptionBotTurnStartStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

AGGREGATION_TIMEOUT = 0.1
USER_TURN_TIMEOUT = 0.2


class TestTranscriptionBotTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_ste(self):
        strategy = TranscriptionBotTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes in between user started/stopped and there are not
        # interim, we just trigger bot speech.
        self.assertTrue(should_start)

    async def test_site(self):
        strategy = TranscriptionBotTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes in between user started/stopped, so we trigger
        # speech right away.
        self.assertTrue(should_start)

    async def test_st1iest2e(self):
        strategy = TranscriptionBotTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # There was an interim before the first user stopped speaking, then we
        # got a transcription comes in between user started/stopped, so we
        # trigger speech right away.
        self.assertTrue(should_start)

    async def test_siet(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_sieit(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_set(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_seit(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_st1et2(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes between user start/stopped speaking, we need to
        # trigger speech right away.
        self.assertTrue(should_start)
        should_start = None

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_set1t2(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_siet1it2(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_t(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_it(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_sie_delay_it(self):
        strategy = TranscriptionBotTurnStartStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Delay
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)


class TestTimeoutBotTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_timeout_user_speaking(self):
        strategy = TimeoutBotTurnStartStrategy(timeout=USER_TURN_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(USER_TURN_TIMEOUT + 0.1)
        self.assertFalse(should_start)

    async def test_timeout_user_not_speaking(self):
        strategy = TimeoutBotTurnStartStrategy(timeout=USER_TURN_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_start = None

        @strategy.event_handler("on_bot_turn_started")
        async def on_bot_turn_started(strategy):
            nonlocal should_start
            should_start = True

        await asyncio.sleep(USER_TURN_TIMEOUT + 0.1)
        self.assertTrue(should_start)
