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


class TestMinWordsCJKCounting(unittest.IsolatedAsyncioTestCase):
    async def test_cjk_transcript_counts_characters(self):
        """Chinese/Japanese text has no spaces: each character is a word (pipecat#6402)."""
        strategy = MinWordsUserTurnStartStrategy(min_words=3)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="你好", user_id="cat", timestamp=""))
        self.assertFalse(should_start)  # 2 chars < 3 words

        await strategy.process_frame(TranscriptionFrame(text="你好呀", user_id="cat", timestamp=""))
        self.assertTrue(should_start)  # 3 chars >= 3 words

    async def test_cjk_two_characters_meet_min_words_threshold(self):
        """Two CJK characters meet min_words=2 while the bot is speaking.

        With the old whitespace counting this transcript was one "word" and the
        interruption never fired (pipecat#6402).
        """
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(text="停下", user_id="cat", timestamp="")  # "stop it"
        )
        self.assertTrue(should_start)

    async def test_korean_still_counts_by_spaces(self):
        """Korean uses eojeol spacing: whitespace counting stays correct."""
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(text="안녕", user_id="cat", timestamp="")  # one word
        )
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="안녕 하세요", user_id="cat", timestamp="")  # two words
        )
        self.assertTrue(should_start)

    async def test_mixed_language_counts_both_channels(self):
        """Mixed CJK + spaced text counts CJK chars and whitespace words together."""
        strategy = MinWordsUserTurnStartStrategy(min_words=4)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        # 2 CJK chars + "hello there" (2 words) = 4
        await strategy.process_frame(
            TranscriptionFrame(text="你好 hello there", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)
