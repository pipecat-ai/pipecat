#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)


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

        # A new turn starts; the strategy re-arms.
        should_start = None
        await strategy.handle_user_turn_started()

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


class TestExternalUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def _capture_params(self, strategy):
        captured = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            captured.append(params)

        return captured

    async def test_external_strategy(self):
        strategy = ExternalUserTurnStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(captured)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(captured)

    async def test_proposal_starts_the_turn_with_emission_enabled(self):
        strategy = ExternalUserTurnStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.process_frame(ProposedUserStartedSpeakingFrame())
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].enable_user_speaking_frames)
        self.assertTrue(captured[0].enable_interruptions)

    async def test_real_turn_frame_starts_the_turn_with_emission_suppressed(self):
        strategy = ExternalUserTurnStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0].enable_user_speaking_frames)
        self.assertFalse(captured[0].enable_interruptions)

    async def test_configured_flags_apply_to_proposals_only(self):
        """Construction settings shape the decide path; the adopt path always suppresses."""
        strategy = ExternalUserTurnStartStrategy(enable_interruptions=False)
        captured = await self._capture_params(strategy)

        await strategy.process_frame(ProposedUserStartedSpeakingFrame())
        self.assertFalse(captured[0].enable_interruptions)
        self.assertTrue(captured[0].enable_user_speaking_frames)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertFalse(captured[1].enable_interruptions)
        self.assertFalse(captured[1].enable_user_speaking_frames)


class TestBaseUserTurnStartStrategyDeprecations(unittest.IsolatedAsyncioTestCase):
    async def _capture_params(self, strategy):
        captured = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            captured.append(params)

        return captured

    async def test_enable_user_speaking_frames_warns(self):
        with self.assertWarns(DeprecationWarning) as caught:
            BaseUserTurnStartStrategy(enable_user_speaking_frames=False)
        self.assertIn("enable_user_speaking_frames", str(caught.warning))

    async def test_enable_user_speaking_frames_applies(self):
        with self.assertWarns(DeprecationWarning):
            strategy = BaseUserTurnStartStrategy(enable_user_speaking_frames=False)
        captured = await self._capture_params(strategy)

        await strategy.trigger_user_turn_started()
        self.assertFalse(captured[0].enable_user_speaking_frames)

    async def test_omitting_enable_user_speaking_frames_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            strategy = BaseUserTurnStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.trigger_user_turn_started()
        self.assertTrue(captured[0].enable_user_speaking_frames)


if __name__ == "__main__":
    unittest.main()
