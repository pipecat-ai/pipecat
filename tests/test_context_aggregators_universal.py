#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantThoughtMessage,
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import FirstSpeechUserMuteStrategy, FunctionCallUserMuteStrategy
from pipecat.turns.user_stop import TranscriptionUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestLLMUserAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_llm_run(self):
        context = LLMContext()

        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [LLMRunFrame()]
        expected_down_frames = [LLMContextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_llm_messages_append(self):
        context = LLMContext()

        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Hi there!",
                    }
                ]
            )
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_append_run(self):
        context = LLMContext()
        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Hi there!",
                    }
                ],
                run_llm=True,
            )
        ]
        expected_down_frames = [LLMContextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_update(self):
        context = LLMContext()
        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            LLMMessagesUpdateFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Hi there!",
                    }
                ]
            )
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_update_run(self):
        context = LLMContext()
        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            LLMMessagesUpdateFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Hi there!",
                    }
                ],
                run_llm=True,
            )
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_default_user_turn_strategies(self):
        context = LLMContext()
        user_aggregator = LLMUserAggregator(context)

        should_start = None
        should_stop = None
        stop_message = None

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            nonlocal should_start
            should_start = True

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message):
            nonlocal should_stop, stop_message
            should_stop = True
            stop_message = message

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InterruptionFrame,
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
            LLMContextFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertEqual(stop_message.content, "Hello!")

    async def test_user_turn_stop_timeout_no_transcription(self):
        context = LLMContext()

        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT),
        )

        should_start = None
        should_stop = None
        timeout = None

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            nonlocal should_start
            should_start = True

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message):
            nonlocal should_stop
            should_stop = True

        @user_aggregator.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(aggregator):
            nonlocal timeout
            timeout = True

        pipeline = Pipeline([user_aggregator])

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
        context = LLMContext()

        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[TranscriptionUserTurnStopStrategy(timeout=TRANSCRIPTION_TIMEOUT)],
                ),
                user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
            ),
        )

        should_start = None
        should_stop = None
        stop_message = None
        timeout = None

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            nonlocal should_start
            should_start = True

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message):
            nonlocal should_stop, stop_message
            should_stop = True
            stop_message = message

        @user_aggregator.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(aggregator):
            nonlocal timeout
            timeout = True

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT - 0.1),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT - 0.1),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT),
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        # The transcription strategy should kick-in before the user turn end timeout.
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertEqual(stop_message.content, "Hello!")
        self.assertFalse(timeout)

    async def test_user_mute_strategies(self):
        context = LLMContext()

        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_mute_strategies=[
                    FirstSpeechUserMuteStrategy(),
                    FunctionCallUserMuteStrategy(),
                ]
            ),
        )

        user_turn = False

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            nonlocal user_turn
            user_turn = True

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            # Bot is speaking, user should be muted.
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
            # Function call is executing, user should be muted.
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        function_name="fn_1", tool_call_id="1", arguments={}, context=None
                    )
                ]
            ),
            SleepFrame(),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            FunctionCallResultFrame(
                function_name="fn_1", tool_call_id="1", arguments={}, result={}
            ),
            SleepFrame(),
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        # The user mute strategies should have muted the user.
        self.assertFalse(user_turn)


class TestLLMAssistantAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_empty(self):
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        should_start = None
        should_stop = None
        stop_message = None

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal should_start
            should_start = True

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            nonlocal should_stop, stop_message
            should_stop = True
            stop_message = message

        frames_to_send = [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]
        await run_test(aggregator, frames_to_send=frames_to_send)
        self.assertTrue(should_start)
        self.assertIsNone(should_stop)
        self.assertIsNone(stop_message)

    async def test_simple(self):
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        should_start = None
        should_stop = None
        stop_message = None

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal should_start
            should_start = True

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            nonlocal should_stop, stop_message
            should_stop = True
            stop_message = message

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello from Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMContextFrame, LLMContextAssistantTimestampFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertEqual(stop_message.content, "Hello from Pipecat!")

    async def test_multiple(self):
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        should_start = None
        should_stop = None
        stop_message = None

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal should_start
            should_start = True

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            nonlocal should_stop, stop_message
            should_stop = True
            stop_message = message

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("from "),
            LLMTextFrame("Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMContextFrame, LLMContextAssistantTimestampFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertEqual(stop_message.content, "Hello from Pipecat!")

    async def test_interruption(self):
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        should_start = 0
        should_stop = 0
        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal should_start
            should_start += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            nonlocal should_stop, stop_messages
            should_stop += 1
            stop_messages.append(message)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            InterruptionFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(should_start, 2)
        self.assertEqual(should_stop, 2)
        self.assertEqual(stop_messages[0].content, "Hello")
        self.assertEqual(stop_messages[1].content, "Hello there!")

    async def test_thought(self):
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        thought_message = None

        @aggregator.event_handler("on_assistant_thought")
        async def on_assistant_thought(aggregator, message: AssistantThoughtMessage):
            nonlocal thought_message
            thought_message = message

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMThoughtStartFrame(),
            LLMThoughtTextFrame(text="I'm thinking!"),
            LLMThoughtEndFrame(),
            LLMFullResponseEndFrame(),
        ]
        await run_test(aggregator, frames_to_send=frames_to_send)
        self.assertEqual(thought_message.content, "I'm thinking!")
