#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings
from unittest import mock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    ProposedUserStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
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

        await strategy.report_user_turn_started()
        self.assertFalse(captured[0].enable_user_speaking_frames)

    async def test_omitting_enable_user_speaking_frames_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            strategy = BaseUserTurnStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.report_user_turn_started()
        self.assertTrue(captured[0].enable_user_speaking_frames)

    async def test_an_override_of_the_deprecated_trigger_still_gets_reports(self):
        with self.assertWarns(DeprecationWarning):

            class OldStartStrategy(BaseUserTurnStartStrategy):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                async def trigger_user_turn_started(self, **kwargs):
                    self.calls += 1
                    await super().trigger_user_turn_started(**kwargs)

        strategy = OldStartStrategy()
        captured = await self._capture_params(strategy)

        await strategy.report_user_turn_started(enable_interruptions=False)

        self.assertEqual(strategy.calls, 1)
        self.assertEqual([p.enable_interruptions for p in captured], [False])

    async def test_a_half_migrated_override_reports_once(self):
        """An override of the old method that calls the new one through super()."""
        with self.assertWarns(DeprecationWarning):

            class HalfMigrated(BaseUserTurnStartStrategy):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                async def trigger_user_turn_started(self, **kwargs):
                    self.calls += 1
                    await super().report_user_turn_started(**kwargs)

        strategy = HalfMigrated()
        captured = await self._capture_params(strategy)

        await strategy.report_user_turn_started(enable_interruptions=False)

        self.assertEqual(strategy.calls, 1)
        self.assertEqual([p.enable_interruptions for p in captured], [False])


if __name__ == "__main__":
    unittest.main()


class TestMinWordsBotPending(unittest.IsolatedAsyncioTestCase):
    """``bot_pending_secs`` applies ``min_words`` before the bot starts speaking."""

    def _strategy(self, **kwargs):
        strategy = MinWordsUserTurnStartStrategy(min_words=2, **kwargs)
        self.started = False

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            self.started = True

        return strategy

    async def _say(self, strategy, text):
        await strategy.process_frame(TranscriptionFrame(text=text, user_id="cat", timestamp=""))

    async def test_disabled_by_default(self):
        strategy = self._strategy()
        await strategy.process_frame(LLMRunFrame())
        await self._say(strategy, "Hello")
        self.assertTrue(self.started)

    async def test_requested_response_needs_min_words(self):
        for request in (
            LLMRunFrame(),
            LLMContextFrame(context=LLMContext()),
            LLMMessagesAppendFrame(messages=[], run_llm=True),
        ):
            with self.subTest(request=type(request).__name__):
                strategy = self._strategy(bot_pending_secs=5.0)
                await strategy.process_frame(request)
                await self._say(strategy, "Hello")
                self.assertFalse(self.started)

                await self._say(strategy, "Hello there")
                self.assertTrue(self.started)

    async def test_append_without_run_llm_is_not_a_request(self):
        strategy = self._strategy(bot_pending_secs=5.0)
        await strategy.process_frame(LLMMessagesAppendFrame(messages=[], run_llm=False))
        await self._say(strategy, "Hello")
        self.assertTrue(self.started)

    async def test_pending_expires(self):
        strategy = self._strategy(bot_pending_secs=5.0)
        with mock.patch("time.monotonic", return_value=100.0):
            await strategy.process_frame(LLMRunFrame())
        with mock.patch("time.monotonic", return_value=105.0):
            await self._say(strategy, "Hello")
        self.assertTrue(self.started)

    async def test_bot_stopped_speaking_ends_pending(self):
        strategy = self._strategy(bot_pending_secs=5.0)
        await strategy.process_frame(LLMRunFrame())
        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(BotStoppedSpeakingFrame())
        await self._say(strategy, "Hello")
        self.assertTrue(self.started)

    async def test_user_turn_started_ends_pending(self):
        strategy = self._strategy(bot_pending_secs=5.0)
        await strategy.process_frame(LLMRunFrame())
        await strategy.handle_user_turn_started()
        await self._say(strategy, "Hello")
        self.assertTrue(self.started)
