#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    TextFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_filter import WakePhraseUserFrameFilter
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

TRANSCRIPTION_TIMEOUT = 0.1
WAKE_TIMEOUT = 0.3


def _make_transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="user1", timestamp="now")


class TestWakePhraseMatching(unittest.IsolatedAsyncioTestCase):
    """Test wake phrase detection logic in isolation via the aggregator."""

    async def test_basic_match(self):
        """Wake phrase in a single transcription frame passes through."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.LISTENING)

        result = await wake_filter.process_frame(_make_transcription("hey robot"))
        self.assertTrue(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_case_insensitive(self):
        """Wake phrase matching is case-insensitive."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        result = await wake_filter.process_frame(_make_transcription("HEY ROBOT"))
        self.assertTrue(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_wake_phrase_within_sentence(self):
        """Wake phrase embedded in a longer transcription is detected."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        result = await wake_filter.process_frame(
            _make_transcription("so I said hey robot what time is it")
        )
        self.assertTrue(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_multiple_phrases(self):
        """Any of multiple configured phrases can trigger activation."""
        wake_filter = WakePhraseUserFrameFilter(
            phrases=["hey robot", "ok computer"], timeout=WAKE_TIMEOUT
        )

        result = await wake_filter.process_frame(_make_transcription("ok computer"))
        self.assertTrue(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_punctuation_stripped(self):
        """Common punctuation is stripped before matching."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        result = await wake_filter.process_frame(_make_transcription("hey, robot!"))
        self.assertTrue(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_no_match_blocks(self):
        """Transcription without wake phrase is blocked."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        result = await wake_filter.process_frame(_make_transcription("hello world"))
        self.assertFalse(result)
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.LISTENING)


class TestWakePhraseBlocking(unittest.IsolatedAsyncioTestCase):
    """Test that frames are correctly blocked/passed in each state."""

    async def test_listening_blocks_vad_frames(self):
        """VAD frames are blocked while LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        self.assertFalse(await wake_filter.process_frame(VADUserStartedSpeakingFrame()))
        self.assertFalse(await wake_filter.process_frame(VADUserStoppedSpeakingFrame()))

    async def test_listening_blocks_interim_transcription(self):
        """Interim transcriptions are blocked while LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        frame = InterimTranscriptionFrame(text="hey", user_id="user1", timestamp="now")
        self.assertFalse(await wake_filter.process_frame(frame))

    async def test_listening_blocks_interruption(self):
        """InterruptionFrame is blocked while LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        self.assertFalse(await wake_filter.process_frame(InterruptionFrame()))

    async def test_listening_passes_user_speaking(self):
        """UserSpeakingFrame passes through while LISTENING (activity frame, not gated)."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        self.assertTrue(await wake_filter.process_frame(UserSpeakingFrame()))

    async def test_listening_passes_other_frames(self):
        """Non-user-interaction frames pass through while LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        self.assertTrue(await wake_filter.process_frame(TextFrame(text="hello")))

    async def test_inactive_passes_all_frames(self):
        """All frames pass through while INACTIVE."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        # Activate
        await wake_filter.process_frame(_make_transcription("hey robot"))
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

        # All frame types should pass
        self.assertTrue(await wake_filter.process_frame(_make_transcription("hello")))
        self.assertTrue(await wake_filter.process_frame(VADUserStartedSpeakingFrame()))
        self.assertTrue(await wake_filter.process_frame(VADUserStoppedSpeakingFrame()))
        self.assertTrue(
            await wake_filter.process_frame(
                InterimTranscriptionFrame(text="hi", user_id="user1", timestamp="now")
            )
        )
        self.assertTrue(await wake_filter.process_frame(InterruptionFrame()))
        self.assertTrue(await wake_filter.process_frame(UserSpeakingFrame()))


class TestWakePhraseTimeout(unittest.IsolatedAsyncioTestCase):
    """Test timeout behavior."""

    async def test_timeout_returns_to_listening(self):
        """Filter returns to LISTENING after inactivity timeout."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            _make_transcription("hey robot"),
            # Wait for timeout to expire
            SleepFrame(sleep=WAKE_TIMEOUT + 0.2),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.LISTENING)

    async def test_activity_refreshes_timeout(self):
        """Speaking activity while INACTIVE refreshes the timeout."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            _make_transcription("hey robot"),
            # Wait just under timeout
            SleepFrame(sleep=WAKE_TIMEOUT * 0.7),
            # UserSpeakingFrame activity should refresh the timeout
            UserSpeakingFrame(),
            # Wait just under timeout again
            SleepFrame(sleep=WAKE_TIMEOUT * 0.7),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        # Should still be INACTIVE because activity refreshed timeout
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

    async def test_timeout_event_fires(self):
        """on_wake_phrase_timeout event fires when timeout expires."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        timeout_fired = False

        @wake_filter.event_handler("on_wake_phrase_timeout")
        async def on_timeout(filter):
            nonlocal timeout_fired
            timeout_fired = True

        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )
        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            _make_transcription("hey robot"),
            SleepFrame(sleep=WAKE_TIMEOUT + 0.2),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertTrue(timeout_fired)


class TestWakePhraseSingleActivation(unittest.IsolatedAsyncioTestCase):
    """Test single_activation mode."""

    async def test_single_activation_resets_on_turn_stop(self):
        """With single_activation, reset() transitions back to LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(
            phrases=["hey robot"], timeout=WAKE_TIMEOUT, single_activation=True
        )

        # Activate
        await wake_filter.process_frame(_make_transcription("hey robot"))
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

        # Simulate turn stop reset
        await wake_filter.reset()
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.LISTENING)

    async def test_default_mode_does_not_reset(self):
        """Without single_activation, reset() does not change state."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)

        # Activate
        await wake_filter.process_frame(_make_transcription("hey robot"))
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)

        # Simulate turn stop reset
        await wake_filter.reset()
        self.assertEqual(wake_filter.state, WakePhraseUserFrameFilter.State.INACTIVE)


class TestWakePhraseEvents(unittest.IsolatedAsyncioTestCase):
    """Test event handlers."""

    async def test_detected_event_fires(self):
        """on_wake_phrase_detected fires with matched phrase text."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        detected_phrase = None

        @wake_filter.event_handler("on_wake_phrase_detected")
        async def on_detected(filter, phrase):
            nonlocal detected_phrase
            detected_phrase = phrase

        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )
        pipeline = Pipeline([user_aggregator])

        frames_to_send = [_make_transcription("hey robot")]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertIsNotNone(detected_phrase)
        self.assertIn("hey", detected_phrase.lower())
        self.assertIn("robot", detected_phrase.lower())


class TestWakePhrasePipelineIntegration(unittest.IsolatedAsyncioTestCase):
    """Test WakePhraseUserFrameFilter in a full pipeline with aggregator."""

    async def test_transcriptions_blocked_while_listening(self):
        """Transcriptions are consumed (not pushed downstream) while LISTENING."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )
        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            # These should all be blocked (no wake phrase)
            VADUserStartedSpeakingFrame(),
            _make_transcription("hello world"),
            VADUserStoppedSpeakingFrame(),
        ]
        # Nothing should come through downstream (transcriptions are consumed,
        # VAD frames are blocked by the filter)
        expected_down_frames = []
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_full_turn_after_wake_phrase(self):
        """After wake phrase, a full user turn completes with context update.

        The VADUserStartedSpeakingFrame sent before the wake phrase is blocked
        by the filter (LISTENING state). The transcription containing the wake
        phrase passes through and triggers the turn via the default
        TranscriptionUserTurnStartStrategy.
        """
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=1.0)
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_frame_filters=[wake_filter],
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
                    ],
                ),
            ),
        )
        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            # Wake phrase transcription - activates filter AND starts the turn
            _make_transcription("hey robot what is the weather"),
            # Wait for speech timeout to trigger turn stop
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
        ]
        expected_down_frames = [
            # TranscriptionUserTurnStartStrategy triggers turn start
            UserStartedSpeakingFrame,
            InterruptionFrame,
            # SpeechTimeoutUserTurnStopStrategy triggers turn stop
            UserStoppedSpeakingFrame,
            LLMContextFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # The transcription should be in context
        self.assertEqual(len(context.messages), 1)
        self.assertEqual(context.messages[0]["role"], "user")
        self.assertIn("hey robot", context.messages[0]["content"])

    async def test_vad_frames_blocked_before_wake(self):
        """VAD frames before wake phrase don't trigger turn start."""
        wake_filter = WakePhraseUserFrameFilter(phrases=["hey robot"], timeout=WAKE_TIMEOUT)
        context = LLMContext()

        turn_started = False

        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_frame_filters=[wake_filter]),
        )

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            nonlocal turn_started
            turn_started = True

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            # VAD events without wake phrase - should be blocked
            VADUserStartedSpeakingFrame(),
            _make_transcription("hello world"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        # Turn should NOT have started because VAD frames were blocked
        self.assertFalse(turn_started)


if __name__ == "__main__":
    unittest.main()
