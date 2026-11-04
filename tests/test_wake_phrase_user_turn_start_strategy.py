#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    MinWordsUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
    _WakeState,
)
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import (
    StrategyMode,
    UserTurnStrategies,
    UserTurnStrategiesParams,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class TestWakePhraseUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    def _create_strategy(self, **kwargs) -> WakePhraseUserTurnStartStrategy:
        kwargs.setdefault("phrases", ["hey pipecat"])
        kwargs.setdefault("timeout", 10.0)
        return WakePhraseUserTurnStartStrategy(**kwargs)

    async def _setup_strategy(self, strategy: WakePhraseUserTurnStartStrategy):
        task_manager = TaskManager()
        await strategy.setup(frame_processor_setup(task_manager))
        # The tests are quick, so make sure the schedule starts all tasks.
        await asyncio.sleep(0)
        return task_manager

    async def test_wake_phrase_in_final_transcription(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_interim_transcription_ignored(self):
        """Interim transcriptions are never used for wake phrase matching."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        await strategy.cleanup()

    async def test_no_wake_phrase_returns_stop(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hello world", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        await strategy.cleanup()

    async def test_non_matching_text_resets_aggregation(self):
        """Non-matching transcription triggers aggregation reset to prevent LLM context pollution."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        reset_called = False

        @strategy.event_handler("on_reset_aggregation")
        async def on_reset_aggregation(strategy):
            nonlocal reset_called
            reset_called = True

        await strategy.process_frame(
            TranscriptionFrame(text="hello world", user_id="user1", timestamp="")
        )
        self.assertTrue(reset_called)

        await strategy.cleanup()

    async def test_vad_frame_returns_stop_in_listening(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        await strategy.cleanup()

    async def test_inactive_returns_continue(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        # Trigger wake phrase first.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Subsequent frames should return CONTINUE.
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(result)

        result = await strategy.process_frame(
            TranscriptionFrame(text="what is the weather", user_id="user1", timestamp="")
        )
        self.assertIsNone(result)

        await strategy.cleanup()

    async def test_accumulation_across_frames(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hey", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        result = await strategy.process_frame(
            TranscriptionFrame(text="pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_multiple_phrases(self):
        strategy = self._create_strategy(phrases=["hey pipecat", "ok computer"])
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="ok computer", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_punctuation_stripped(self):
        """STT punctuation like 'Hey, Pipecat!' should still match."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="Hey, Pipecat!", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_turn_start_preserves_inactive_state(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.handle_user_turn_started()
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_timeout_returns_to_listening(self):
        strategy = self._create_strategy(timeout=0.1)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Wait for timeout to expire.
        await asyncio.sleep(0.3)

        self.assertEqual(strategy.state, _WakeState.IDLE)

        await strategy.cleanup()

    async def test_activity_refreshes_timeout(self):
        strategy = self._create_strategy(timeout=0.2)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Send activity before timeout.
        await asyncio.sleep(0.1)
        await strategy.process_frame(UserSpeakingFrame())
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Send more activity.
        await asyncio.sleep(0.1)
        await strategy.process_frame(BotSpeakingFrame())
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Wait for timeout to expire after last activity.
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        await strategy.cleanup()

    async def test_wake_phrase_detected_event(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        detected_phrase = None

        @strategy.event_handler("on_wake_phrase_detected")
        async def on_wake_phrase_detected(strategy, phrase):
            nonlocal detected_phrase
            detected_phrase = phrase

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )

        # Event fires in a background task, give it a moment.
        await asyncio.sleep(0.05)
        self.assertEqual(detected_phrase, "hey pipecat")

        await strategy.cleanup()

    async def test_wake_phrase_timeout_event(self):
        strategy = self._create_strategy(timeout=0.1)
        await self._setup_strategy(strategy)

        timeout_fired = False

        @strategy.event_handler("on_wake_phrase_timeout")
        async def on_wake_phrase_timeout(strategy):
            nonlocal timeout_fired
            timeout_fired = True

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )

        # Wait for timeout.
        await asyncio.sleep(0.3)
        self.assertTrue(timeout_fired)

        await strategy.cleanup()

    async def test_single_activation_stays_inactive_after_turn_start(self):
        """In single activation mode, the turn-start callback keeps INACTIVE so the current turn can finish."""
        strategy = self._create_strategy(single_activation=True, timeout=0.5)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Simulate turn start (the controller notifies all start strategies).
        await strategy.handle_user_turn_started()
        # State remains INACTIVE so frames continue to flow.
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Subsequent frames should pass through (CONTINUE).
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(result)

        result = await strategy.process_frame(
            TranscriptionFrame(text="what is the weather", user_id="user1", timestamp="")
        )
        self.assertIsNone(result)

        await strategy.cleanup()

    async def test_single_activation_timeout_returns_to_listening(self):
        """In single activation mode, the keepalive timeout returns to LISTENING."""
        strategy = self._create_strategy(single_activation=True, timeout=0.1)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        # Wait for keepalive timeout to expire.
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        # Frames should now be blocked again.
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.STOP)

        await strategy.cleanup()

    async def test_single_activation_requires_wake_phrase_after_timeout(self):
        """Single activation mode requires wake phrase again after keepalive timeout."""
        strategy = self._create_strategy(single_activation=True, timeout=0.1)
        await self._setup_strategy(strategy)

        # First turn: wake phrase -> INACTIVE -> timeout -> LISTENING.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.IDLE)

        # Without wake phrase, frames are blocked.
        result = await strategy.process_frame(
            TranscriptionFrame(text="what is the weather", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)

        # Second turn: wake phrase again.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()


class EndOnDemandStopStrategy(BaseUserTurnStopStrategy):
    """A stop strategy that ends the turn when a test says so."""

    async def end(self):
        await self.report_user_turn_stopped()


class TestWakePhraseWithAllStartMode(unittest.IsolatedAsyncioTestCase):
    """With StrategyMode.ALL, a turn needs the user speaking and the wake phrase."""

    async def _start(self, *, wake_first: bool = False, timeout: float = 10.0):
        self.wake = WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"], timeout=timeout)
        vad = VADUserTurnStartStrategy()
        start: list[BaseUserTurnStartStrategy] = (
            [self.wake, vad] if wake_first else [vad, self.wake]
        )
        self.stop = EndOnDemandStopStrategy()
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=start, stop=[self.stop]),
            user_turn_strategies_params=UserTurnStrategiesParams(start_mode=StrategyMode.ALL),
        )
        await self.controller.setup(frame_processor_setup(TaskManager()))
        self.starts = 0

        @self.controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            self.starts += 1

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _say(self, text: str):
        frames: list[Frame] = [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            TranscriptionFrame(text=text, user_id="user1", timestamp=""),
        ]
        for frame in frames:
            await self.controller.process_frame(frame)

    async def test_speech_without_the_wake_phrase_starts_no_turn(self):
        await self._start()

        await self._say("what is the weather")

        self.assertEqual(self.starts, 0)

    async def test_the_wake_phrase_starts_a_turn_in_either_order(self):
        for wake_first in (False, True):
            with self.subTest(wake_first=wake_first):
                await self._start(wake_first=wake_first)

                await self._say("hey pipecat what is the weather")

                self.assertEqual(self.starts, 1)
                await self.controller.cleanup()

    async def test_speech_starts_a_turn_while_awake(self):
        await self._start()
        await self._say("hey pipecat")
        await self.stop.end()

        await self._say("what is the weather")

        self.assertEqual(self.starts, 2)

    async def test_speech_starts_no_turn_once_the_wake_phrase_times_out(self):
        await self._start(timeout=0.05)
        await self._say("hey pipecat")
        await self.stop.end()
        await asyncio.sleep(0.15)

        await self._say("what is the weather")

        self.assertEqual(self.wake.state, _WakeState.IDLE)
        self.assertEqual(self.starts, 1)


class TestWakePhraseAlone(unittest.IsolatedAsyncioTestCase):
    """On its own, the wake phrase strategy starts turns, even on speech VAD missed."""

    async def asyncSetUp(self):
        self.wake = WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"])
        self.stop = EndOnDemandStopStrategy()
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[self.wake], stop=[self.stop]),
        )
        await self.controller.setup(frame_processor_setup(TaskManager()))
        self.starts = 0

        @self.controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            self.starts += 1

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _transcribe(self, text: str):
        await self.controller.process_frame(
            TranscriptionFrame(text=text, user_id="user1", timestamp="")
        )

    async def test_speech_without_the_wake_phrase_starts_no_turn(self):
        await self._transcribe("what is the weather")

        self.assertEqual(self.starts, 0)

    async def test_the_wake_phrase_starts_a_turn_without_vad(self):
        await self._transcribe("hey pipecat what is the weather")

        self.assertEqual(self.starts, 1)

    async def test_a_transcription_starts_a_turn_while_awake_without_vad(self):
        await self._transcribe("hey pipecat")
        await self.stop.end()

        await self._transcribe("what is the weather")

        self.assertEqual(self.starts, 2)


class TestWakePhraseWithConditions(unittest.IsolatedAsyncioTestCase):
    """In an ALL list, the wake phrase combines with other start conditions."""

    async def asyncSetUp(self):
        self.stop = EndOnDemandStopStrategy()
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[
                    WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"]),
                    MinWordsUserTurnStartStrategy(min_words=3),
                ],
                stop=[self.stop],
            ),
            user_turn_strategies_params=UserTurnStrategiesParams(start_mode=StrategyMode.ALL),
        )
        await self.controller.setup(frame_processor_setup(TaskManager()))
        self.starts = 0

        @self.controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            self.starts += 1

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _transcribe(self, text: str):
        await self.controller.process_frame(
            TranscriptionFrame(text=text, user_id="user1", timestamp="")
        )

    async def test_only_enough_words_interrupt_the_bot_once_awake(self):
        await self._transcribe("hey pipecat")
        await self.stop.end()
        await self.controller.process_frame(BotStartedSpeakingFrame())

        await self._transcribe("uh huh")
        self.assertEqual(self.starts, 1)

        await self._transcribe("wait a second please")
        self.assertEqual(self.starts, 2)


if __name__ == "__main__":
    unittest.main()
