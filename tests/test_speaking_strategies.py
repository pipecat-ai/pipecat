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
from pipecat.turns.eager_transcription_speaking_strategy import EagerTranscriptionSpeakingStrategy
from pipecat.turns.lazy_transcription_speaking_strategy import LazyTranscriptionSpeakingStrategy
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

AGGREGATION_TIMEOUT = 0.1


class TestEagerTranscriptionSpeakingStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_speaking_strategy(self):
        strategy = EagerTranscriptionSpeakingStrategy()

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # Still speaking
        await strategy.process_frame(
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp="now")
        )
        self.assertIsNone(should_speak)

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="now")
        )
        self.assertTrue(should_speak)


class TestLazyTranscriptionSpeakingStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_ste(self):
        strategy = LazyTranscriptionSpeakingStrategy()

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes in between user started/stopped and there are not
        # interim, we just trigger bot speech.
        self.assertTrue(should_speak)

    async def test_site(self):
        strategy = LazyTranscriptionSpeakingStrategy()

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes in between user started/stopped, so we trigger
        # speech right away.
        self.assertTrue(should_speak)

    async def test_st1iest2e(self):
        strategy = LazyTranscriptionSpeakingStrategy()

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # There was an interim before the first user stopped speaking, then we
        # got a transcription comes in between user started/stopped, so we
        # trigger speech right away.
        self.assertTrue(should_speak)

    async def test_siet(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_sieit(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_set(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_seit(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_st1et2(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Transcription comes between user start/stopped speaking, we need to
        # trigger speech right away.
        self.assertTrue(should_speak)
        should_speak = None

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_set1t2(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_siet1it2(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_speak)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_t(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_it(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)

    async def test_sie_delay_it(self):
        strategy = LazyTranscriptionSpeakingStrategy(timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)

        should_speak = None

        @strategy.event_handler("on_should_speak")
        async def on_should_speak(strategy):
            nonlocal should_speak
            should_speak = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_speak)

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
        self.assertIsNone(should_speak)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_speak)
