#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import Mock

from pipecat.frames.frames import (
    BotOutputAudioPauseFrame,
    BotOutputAudioResumeFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    MinWordsUserTurnStartStrategy,
    ProvisionalVADUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_bot_speaking_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
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

    async def test_bot_speaking_singlw_words(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=3)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="One", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Two", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Three", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

    async def test_bot_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
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
        async def on_user_turn_started(strategy, params):
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
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertTrue(should_start)

    async def test_bot_not_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
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
        async def on_user_turn_started(strategy, params):
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
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        self.assertTrue(should_start)


class TestProvisionalVADUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_vad_while_bot_speaking_pauses_without_starting_turn(self):
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=0.05)
        await strategy.setup(self.task_manager)

        pushed_frames = []
        should_start = False

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertFalse(should_start)
        self.assertIsInstance(pushed_frames[-1], BotOutputAudioPauseFrame)

        await strategy.cleanup()

    async def test_transcript_during_provisional_pause_starts_turn(self):
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=1.0)
        await strategy.setup(self.task_manager)

        pushed_frames = []
        should_start = False

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertTrue(should_start)
        # Confirming the turn resumes output audio explicitly so the transport
        # is unblocked even when interruptions are disabled.
        self.assertEqual(
            [type(frame) for frame in pushed_frames],
            [BotOutputAudioPauseFrame, BotOutputAudioResumeFrame],
        )

        await strategy.cleanup()

    async def test_no_transcript_resumes_and_locks_out_until_bot_stops(self):
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=0.01)
        await strategy.setup(self.task_manager)

        pushed_frames = []
        should_start = False
        reset_aggregation = Mock()

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        @strategy.event_handler("on_reset_aggregation")
        async def on_reset_aggregation(strategy):
            reset_aggregation()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await self._wait_for_frame(pushed_frames, BotOutputAudioResumeFrame)

        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="late", user_id="cat", timestamp="")
        )

        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertFalse(should_start)
        reset_aggregation.assert_called_once()

        await strategy.process_frame(BotStoppedSpeakingFrame())
        result = await strategy.process_frame(
            TranscriptionFrame(text="new", user_id="cat", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertTrue(should_start)

        await strategy.cleanup()

    async def test_reset_during_provisional_pause_resumes_output_audio(self):
        # A long pause window keeps the provisional pause armed so the only way
        # the transport gets resumed is the reset() guard itself.
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=10.0)
        await strategy.setup(self.task_manager)

        pushed_frames = []

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsInstance(pushed_frames[-1], BotOutputAudioPauseFrame)

        await strategy.reset()

        self.assertIsInstance(pushed_frames[-1], BotOutputAudioResumeFrame)

        await strategy.cleanup()

    async def test_cleanup_during_provisional_pause_resumes_output_audio(self):
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=10.0)
        await strategy.setup(self.task_manager)

        pushed_frames = []

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsInstance(pushed_frames[-1], BotOutputAudioPauseFrame)

        await strategy.cleanup()

        self.assertIsInstance(pushed_frames[-1], BotOutputAudioResumeFrame)

    async def test_transcript_resumes_output_audio_when_interruptions_disabled(self):
        # With interruptions disabled, trigger_user_turn_started() emits no
        # InterruptionFrame, so the strategy must resume the transport itself.
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=10.0, enable_interruptions=False)
        await strategy.setup(self.task_manager)

        pushed_frames = []
        should_start = False

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed_frames.append(frame)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertTrue(should_start)
        self.assertEqual(
            [type(frame) for frame in pushed_frames],
            [BotOutputAudioPauseFrame, BotOutputAudioResumeFrame],
        )

        await strategy.cleanup()

    async def _wait_for_frame(self, frames, frame_type):
        for _ in range(20):
            if any(isinstance(frame, frame_type) for frame in frames):
                return
            await asyncio.sleep(0.01)
        self.fail(f"{frame_type.__name__} was not pushed")


class TestExternalUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_external_strategy(self):
        strategy = ExternalUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(should_start)


if __name__ == "__main__":
    unittest.main()
