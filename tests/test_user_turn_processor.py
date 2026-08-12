#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    InterruptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
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
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestUserTurnProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_default_user_turn_strategies(self):
        user_turn_processor = UserTurnProcessor(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        should_start = None
        should_stop = None

        @user_turn_processor.event_handler("on_user_turn_started")
        async def on_user_turn_started(processor, strategy):
            nonlocal should_start
            should_start = True

        @user_turn_processor.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(processor, strategy):
            nonlocal should_stop
            should_stop = True

        pipeline = Pipeline([user_turn_processor])

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            # Wait for user_speech_timeout to elapse
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
        ]
        expected_down_frames = [
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InterruptionFrame,
            TranscriptionFrame,
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)

    async def test_user_turn_stop_timeout_no_transcription(self):
        user_turn_processor = UserTurnProcessor(
            user_turn_strategies=UserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        should_start = None
        should_stop = None
        timeout = None

        @user_turn_processor.event_handler("on_user_turn_started")
        async def on_user_turn_started(processor, strategy):
            nonlocal should_start
            should_start = True

        @user_turn_processor.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(processor, strategy):
            nonlocal should_stop
            should_stop = True

        @user_turn_processor.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(processor):
            nonlocal timeout
            timeout = True

        pipeline = Pipeline([user_turn_processor])

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.1),
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_user_turn_stop_timeout_transcription(self):
        user_turn_processor = UserTurnProcessor(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        should_start = None
        should_stop = None
        timeout = None

        @user_turn_processor.event_handler("on_user_turn_started")
        async def on_user_turn_started(processor, strategy):
            nonlocal should_start
            should_start = True

        @user_turn_processor.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(processor, strategy):
            nonlocal should_stop
            should_stop = True

        @user_turn_processor.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(processor):
            nonlocal timeout
            timeout = True

        pipeline = Pipeline([user_turn_processor])

        # Transcript arrives before VAD stop, then we wait for user_speech_timeout
        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            VADUserStoppedSpeakingFrame(),
            # Wait for user_speech_timeout (TRANSCRIPTION_TIMEOUT=0.1s) to elapse
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.05),
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        # The transcription strategy should kick-in before the user turn end timeout.
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertFalse(timeout)

    async def test_proposal_is_resolved_once_with_an_aggregator_downstream(self):
        """Two resolvers in one pipeline must not decide the same turn twice.

        The processor resolves the proposal and consumes it, so the aggregator
        downstream sees only the turn frames it emitted and adopts them.
        """
        user_turn_processor = UserTurnProcessor(user_turn_strategies=ExternalUserTurnStrategies())
        user_aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
        )

        frames_to_send = [
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=1.0),
        ]
        received_down, _ = await run_test(
            Pipeline([user_turn_processor, user_aggregator]),
            frames_to_send=frames_to_send,
            expected_down_frames=None,
        )

        names = [type(f).__name__ for f in received_down]
        self.assertEqual(names.count("UserStartedSpeakingFrame"), 1)
        self.assertEqual(names.count("UserStoppedSpeakingFrame"), 1)
        self.assertEqual(names.count("InterruptionFrame"), 1)
        self.assertEqual(names.count("ProposedUserStartedSpeakingFrame"), 0)
        self.assertEqual(names.count("ProposedUserStoppedSpeakingFrame"), 0)


if __name__ == "__main__":
    unittest.main()
