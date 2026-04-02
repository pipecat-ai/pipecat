#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TranslationFrame,
    UserMuteStartedFrame,
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
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import (
    FirstSpeechUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestLLMUserAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_llm_run(self):
        context = LLMContext()

        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [LLMRunFrame()]
        expected_down_frames = [SpeechControlParamsFrame, LLMContextFrame]
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
        expected_down_frames = [SpeechControlParamsFrame, LLMContextFrame]
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

    async def test_llm_messages_update_does_not_inject_turn_completion_into_context(self):
        context = LLMContext()
        params = LLMUserAggregatorParams(filter_incomplete_user_turns=True)
        pipeline = Pipeline([LLMUserAggregator(context, params=params)])

        new_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        frames_to_send = [LLMMessagesUpdateFrame(messages=new_messages)]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )
        # Turn completion instructions are now set via system_instruction on the
        # LLM service, not injected into context messages.
        assert len(context.messages) == 2
        assert context.messages[0]["content"] == "You are a helpful assistant."
        assert context.messages[1]["content"] == "Hello!"

    async def test_llm_messages_transform(self):
        context = LLMContext()
        # Set up initial messages
        context.set_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        pipeline = Pipeline([LLMUserAggregator(context)])

        # Transform that keeps only user messages
        def keep_user_messages(messages):
            return [m for m in messages if m["role"] == "user"]

        frames_to_send = [LLMMessagesTransformFrame(transform=keep_user_messages)]
        expected_down_frames = [
            SpeechControlParamsFrame  # no LLMContextFrame expected, run_llm defaults to False
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert len(context.messages) == 2
        assert context.messages[0]["content"] == "Hello"
        assert context.messages[1]["content"] == "How are you?"

    async def test_llm_messages_transform_run(self):
        context = LLMContext()
        # Set up initial messages
        context.set_messages([{"role": "user", "content": "Hello"}])

        pipeline = Pipeline([LLMUserAggregator(context)])

        # Transform that modifies the content
        def uppercase_content(messages):
            return [{"role": m["role"], "content": m["content"].upper()} for m in messages]

        frames_to_send = [LLMMessagesTransformFrame(transform=uppercase_content, run_llm=True)]
        expected_down_frames = [SpeechControlParamsFrame, LLMContextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == "HELLO"

    async def test_default_user_turn_strategies(self):
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
                    ],
                ),
            ),
        )

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
            # Wait for user_speech_timeout to elapse
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
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
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
                    ],
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

    async def test_pending_transcription_emitted_on_end_frame(self):
        """Pending user transcription should be emitted when EndFrame arrives."""
        context = LLMContext()

        user_aggregator = LLMUserAggregator(context)

        stop_messages = []

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message):
            stop_messages.append((strategy, message))

        pipeline = Pipeline([user_aggregator])

        # Start turn and send transcription, but don't trigger normal turn stop
        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            # No VADUserStoppedSpeakingFrame - turn doesn't stop normally
            # EndFrame will be sent by run_test, triggering emission
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        # The pending transcription should be emitted on EndFrame
        self.assertEqual(len(stop_messages), 1)
        strategy, message = stop_messages[0]
        self.assertIsNone(strategy)  # strategy is None for end/cancel
        self.assertEqual(message.content, "Hello!")

    async def test_start_frame_before_mute_event(self):
        """StartFrame must reach downstream before mute events are broadcast.

        With MuteUntilFirstBotCompleteUserMuteStrategy, the mute logic should
        not run on control frames (StartFrame, EndFrame, CancelFrame). This
        ensures StartFrame reaches downstream processors before
        UserMuteStartedFrame is broadcast.

        The default TurnAnalyzerUserTurnStopStrategy broadcasts a
        SpeechControlParamsFrame when it processes StartFrame, which gets
        re-queued to the aggregator. That non-control frame legitimately
        triggers the mute state change, so UserMuteStartedFrame follows
        StartFrame — but crucially, after it.
        """
        context = LLMContext()

        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            ),
        )

        pipeline = Pipeline([user_aggregator])

        # run_test internally sends StartFrame via PipelineRunner. With
        # ignore_start=False we can verify ordering: StartFrame must arrive
        # before UserMuteStartedFrame. Before the fix, UserMuteStartedFrame
        # was broadcast before StartFrame reached downstream processors.
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=[],
            expected_down_frames=[StartFrame, UserMuteStartedFrame, SpeechControlParamsFrame],
            ignore_start=False,
        )

    async def test_interim_transcription_not_pushed_downstream(self):
        """InterimTranscriptionFrame should be consumed and not pushed downstream."""
        context = LLMContext()
        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            InterimTranscriptionFrame(text="Hel", user_id="", timestamp="now"),
            InterimTranscriptionFrame(text="Hello", user_id="", timestamp="now"),
        ]
        # The interim transcription triggers a user turn start via the default
        # TranscriptionUserTurnStartStrategy, so we expect turn-related frames
        # but NOT the InterimTranscriptionFrame itself.
        expected_down_frames = [
            SpeechControlParamsFrame,
            UserStartedSpeakingFrame,
            InterruptionFrame,
        ]
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertFalse(any(isinstance(f, InterimTranscriptionFrame) for f in down_frames))

    async def test_translation_not_pushed_downstream(self):
        """TranslationFrame should be consumed and not pushed downstream."""
        context = LLMContext()
        pipeline = Pipeline([LLMUserAggregator(context)])

        frames_to_send = [
            TranslationFrame(text="Hola!", user_id="", timestamp="now", language="es"),
        ]
        # Only the SpeechControlParamsFrame from the default turn strategy on
        # start is expected — the translation itself is consumed.
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=[SpeechControlParamsFrame],
        )


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

    async def test_multiple_text_with_spaces(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        def make_text_frame(text: str) -> TextFrame:
            frame = TextFrame(text=text)
            frame.includes_inter_frame_spaces = True
            return frame

        frames_to_send = [
            LLMFullResponseStartFrame(),
            make_text_frame("Hello "),
            make_text_frame("Pipecat. "),
            make_text_frame("How are "),
            make_text_frame("you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMContextFrame, LLMContextAssistantTimestampFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == "Hello Pipecat. How are you?"

    async def test_multiple_text_stripped(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello"),
            TextFrame(text="Pipecat."),
            TextFrame(text="How are"),
            TextFrame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMContextFrame, LLMContextAssistantTimestampFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == "Hello Pipecat. How are you?"

    async def test_multiple_text_mixed_spaces(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        def make_text_frame(text: str, includes_spaces: bool) -> TextFrame:
            frame = TextFrame(text=text)
            frame.includes_inter_frame_spaces = includes_spaces
            return frame

        frames_to_send = [
            LLMFullResponseStartFrame(),
            make_text_frame("Hello ", includes_spaces=True),
            make_text_frame("Pipecat. ", includes_spaces=True),
            make_text_frame("Here's some", includes_spaces=True),
            make_text_frame(
                " code:", includes_spaces=True
            ),  # Validates ending includes_inter_frame_spaces run with no space
            make_text_frame("```python\nprint('Hello, World!')\n```", includes_spaces=False),
            make_text_frame(
                "```javascript\nconsole.log('Hello, World!');\n```", includes_spaces=False
            ),
            make_text_frame(
                " And some more: ", includes_spaces=True
            ),  # Validates starting includes_inter_frame_spaces run with a space and ending it with no space
            make_text_frame("```html\n<div>Hello, World!</div>\n```", includes_spaces=False),
            make_text_frame(
                "Hope that ", includes_spaces=True
            ),  # Validates starting includes_inter_frame_spaces run with no space
            make_text_frame("helps!", includes_spaces=True),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMContextFrame, LLMContextAssistantTimestampFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == (
            "Hello Pipecat. Here's some code: "
            "```python\nprint('Hello, World!')\n``` "
            "```javascript\nconsole.log('Hello, World!');\n``` "
            "And some more: "
            "```html\n<div>Hello, World!</div>\n``` "
            "Hope that helps!"
        )

    async def test_multiple_responses(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        def make_text_frame(text: str) -> TextFrame:
            frame = TextFrame(text=text)
            frame.includes_inter_frame_spaces = True
            return frame

        frames_to_send = [
            LLMFullResponseStartFrame(),
            make_text_frame("Hello "),
            make_text_frame("Pipecat."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            make_text_frame(text="How are "),
            make_text_frame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert context.messages[0]["content"] == "Hello Pipecat."
        assert context.messages[1]["content"] == "How are you?"

    async def test_multiple_responses_interruption(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        def make_text_frame(text: str) -> TextFrame:
            frame = TextFrame(text=text)
            frame.includes_inter_frame_spaces = True
            return frame

        frames_to_send = [
            LLMFullResponseStartFrame(),
            make_text_frame("Hello "),
            make_text_frame("Pipecat."),
            LLMFullResponseEndFrame(),
            SleepFrame(0.15),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            make_text_frame("How are "),
            make_text_frame("you?"),
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
        assert context.messages[0]["content"] == "Hello Pipecat."
        assert context.messages[1]["content"] == "How are you?"

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

    async def test_function_call(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)
        frames_to_send = [
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={"location": "Los Angeles"},
                cancel_on_interruption=False,
            ),
            SleepFrame(),
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={"location": "Los Angeles"},
                result={"conditions": "Sunny"},
            ),
        ]
        expected_down_frames = []
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert json.loads(context.messages[-1]["content"]) == {"conditions": "Sunny"}

    async def test_function_call_on_context_updated(self):
        context_updated = False

        async def on_context_updated():
            nonlocal context_updated
            context_updated = True

        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)
        frames_to_send = [
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={"location": "Los Angeles"},
                cancel_on_interruption=False,
            ),
            SleepFrame(),
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={"location": "Los Angeles"},
                result={"conditions": "Sunny"},
                properties=FunctionCallResultProperties(on_context_updated=on_context_updated),
            ),
            SleepFrame(),
        ]
        expected_down_frames = []
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert json.loads(context.messages[-1]["content"]) == {"conditions": "Sunny"}
        assert context_updated

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

    async def test_pending_text_emitted_on_end_frame(self):
        """Pending assistant text should be emitted when EndFrame arrives."""
        context = LLMContext()

        aggregator = LLMAssistantAggregator(context)

        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        # Start response and send text, but don't send LLMFullResponseEndFrame
        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello from Pipecat!"),
            # No LLMFullResponseEndFrame - response doesn't end normally
            # EndFrame will be sent by run_test, triggering emission
        ]
        await run_test(aggregator, frames_to_send=frames_to_send)

        # The pending text should be emitted on EndFrame
        self.assertEqual(len(stop_messages), 1)
        self.assertEqual(stop_messages[0].content, "Hello from Pipecat!")

    async def test_turn_completion_markers_stripped_from_transcript(self):
        """Turn completion markers should be stripped from assistant transcript."""
        from pipecat.turns.user_turn_completion_mixin import (
            USER_TURN_COMPLETE_MARKER,
            USER_TURN_INCOMPLETE_SHORT_MARKER,
        )

        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        # Send text with a turn completion marker
        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame(f"{USER_TURN_COMPLETE_MARKER} Hello from Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        await run_test(aggregator, frames_to_send=frames_to_send)

        # The marker should be stripped from the transcript
        self.assertEqual(len(stop_messages), 1)
        self.assertEqual(stop_messages[0].content, "Hello from Pipecat!")

        # Test incomplete markers are also stripped
        stop_messages.clear()
        context2 = LLMContext()
        aggregator2 = LLMAssistantAggregator(context2)

        @aggregator2.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped2(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame(USER_TURN_INCOMPLETE_SHORT_MARKER),
            LLMFullResponseEndFrame(),
        ]
        await run_test(aggregator2, frames_to_send=frames_to_send)

        # The incomplete marker should be stripped (resulting in empty content)
        self.assertEqual(len(stop_messages), 1)
        self.assertEqual(stop_messages[0].content, "")

    async def test_llm_run(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        expected_up_frames = [LLMContextFrame]
        await run_test(
            aggregator,
            frames_to_send=[LLMRunFrame()],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=expected_up_frames,
        )

    async def test_llm_messages_append(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi there!",
                        }
                    ]
                )
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_append_run(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        expected_up_frames = [LLMContextFrame]
        await run_test(
            aggregator,
            frames_to_send=[
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi there!",
                        }
                    ],
                    run_llm=True,
                )
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=expected_up_frames,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_update(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                LLMMessagesUpdateFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi there!",
                        }
                    ]
                )
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_update_run(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                LLMMessagesUpdateFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi there!",
                        }
                    ],
                    run_llm=True,
                )
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )
        assert context.messages[0]["content"] == "Hi there!"

    async def test_llm_messages_transform(self):
        context = LLMContext()
        context.set_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        aggregator = LLMAssistantAggregator(context)

        # Transform that keeps only user messages
        def keep_user_messages(messages):
            return [m for m in messages if m["role"] == "user"]

        await run_test(
            aggregator,
            frames_to_send=[LLMMessagesTransformFrame(transform=keep_user_messages)],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )
        assert len(context.messages) == 2
        assert context.messages[0]["content"] == "Hello"
        assert context.messages[1]["content"] == "How are you?"

    async def test_llm_messages_transform_run(self):
        context = LLMContext()
        context.set_messages([{"role": "user", "content": "Hello"}])

        aggregator = LLMAssistantAggregator(context)

        # Transform that modifies the content
        def uppercase_content(messages):
            return [{"role": m["role"], "content": m["content"].upper()} for m in messages]

        expected_up_frames = [LLMContextFrame]
        await run_test(
            aggregator,
            frames_to_send=[LLMMessagesTransformFrame(transform=uppercase_content, run_llm=True)],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=expected_up_frames,
        )
        assert context.messages[0]["content"] == "HELLO"


if __name__ == "__main__":
    unittest.main()
