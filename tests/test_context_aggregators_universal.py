#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
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
    LLMAssistantPushAggregationFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    RealtimeServiceMetadataFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TranslationFrame,
    TTSStartedFrame,
    TTSTextFrame,
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
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import (
    FirstSpeechUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import (
    FilterIncompleteUserTurnStrategies,
    UserTurnStrategies,
)
from pipecat.utils.text.base_text_aggregator import AggregationType

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
        expected_down_frames = [
            SpeechControlParamsFrame  # no LLMContextFrame expected, run_llm defaults to False
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
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
        expected_down_frames = [
            SpeechControlParamsFrame  # no LLMContextFrame expected, run_llm defaults to False
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
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
        params = LLMUserAggregatorParams(
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        )
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
            LLMContextFrame,
            UserStoppedSpeakingFrame,
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

    async def test_function_call_start_force_stops_user_turn_without_llm_rerun(self):
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

        stop_messages = []

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message):
            stop_messages.append((strategy, message))

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        function_name="fn_1", tool_call_id="1", arguments={}, context=None
                    )
                ]
            ),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
        ]
        (down_frames, _) = await run_test(
            Pipeline([user_aggregator]), frames_to_send=frames_to_send
        )

        down_frame_types = [type(frame) for frame in down_frames]

        self.assertIn(UserStoppedSpeakingFrame, down_frame_types)
        self.assertNotIn(LLMContextFrame, down_frame_types)
        self.assertEqual(len(stop_messages), 1)
        self.assertIsNone(stop_messages[0][0])
        self.assertEqual(stop_messages[0][1].content, "Hello!")

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

        # run_test internally sends StartFrame via WorkerRunner. With
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

    async def test_inference_triggered_event_fires_on_default_strategies(self):
        """Default flow fires inference-triggered before stopped, both with the same strategy."""
        from pipecat.frames.frames import UserTurnInferenceCompletedFrame  # noqa: F401

        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
                    ]
                ),
            ),
        )

        events: list[str] = []

        @user_aggregator.event_handler("on_user_turn_inference_triggered")
        async def on_inference_triggered(aggregator, strategy):
            events.append("inference_triggered")

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_stopped(aggregator, strategy, message):
            events.append(f"stopped:{message.content}")

        pipeline = Pipeline([user_aggregator])
        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hi!", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertEqual(events, ["inference_triggered", "stopped:Hi!"])

    async def test_filter_incomplete_user_turns_emits_deprecation_warning(self):
        """Setting the legacy flag emits a DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMUserAggregatorParams(filter_incomplete_user_turns=True)
            matched = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "filter_incomplete_user_turns" in str(x.message)
            ]
            self.assertTrue(matched, "expected a DeprecationWarning")

    async def test_filter_incomplete_user_turns_installs_strategy(self):
        """Legacy flag wraps existing stops with deferred() and appends the LLM strategy."""
        import warnings

        from pipecat.turns.user_stop import (
            DeferredUserTurnStopStrategy,
            LLMTurnCompletionUserTurnStopStrategy,
            SpeechTimeoutUserTurnStopStrategy,
        )

        existing = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)

        context = LLMContext()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            params = LLMUserAggregatorParams(
                filter_incomplete_user_turns=True,
                user_turn_strategies=UserTurnStrategies(stop=[existing]),
            )
            aggregator = LLMUserAggregator(context, params=params)

        stop_strategies = aggregator._params.user_turn_strategies.stop
        self.assertEqual(len(stop_strategies), 2)
        self.assertIsInstance(stop_strategies[0], DeferredUserTurnStopStrategy)
        self.assertIs(stop_strategies[0].inner, existing)
        self.assertIsInstance(stop_strategies[1], LLMTurnCompletionUserTurnStopStrategy)

    async def test_llm_completion_strategy_finalizes_on_complete_marker(self):
        """LLMTurnCompletionUserTurnStopStrategy finalizes only on UserTurnInferenceCompletedFrame(complete)."""
        from pipecat.frames.frames import UserTurnInferenceCompletedFrame
        from pipecat.turns.user_stop import LLMTurnCompletionUserTurnStopStrategy, deferred

        gating = LLMTurnCompletionUserTurnStopStrategy()
        upstream = deferred(
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        )
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(stop=[upstream, gating]),
            ),
        )

        events: list[str] = []

        @user_aggregator.event_handler("on_user_turn_inference_triggered")
        async def on_inference_triggered(aggregator, strategy):
            events.append("inference_triggered")

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_stopped(aggregator, strategy, message):
            events.append("stopped")

        pipeline = Pipeline([user_aggregator])

        # Drive the pipeline. Inference fires after the upstream
        # strategy's timeout. Stop fires only when UserTurnInferenceCompletedFrame
        # arrives (producer absence == "not yet complete").
        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hi", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
            # At this point inference_triggered should have fired but NOT stopped.
            UserTurnInferenceCompletedFrame(),
            SleepFrame(),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertEqual(events, ["inference_triggered", "stopped"])

    async def test_multiple_inferences_in_one_turn_preserve_aggregation(self):
        """Two inference triggers before finalization should preserve the full user transcript.

        When the LLM marks the first inference incomplete (○ / ◐) and the
        user keeps speaking, the deferred upstream strategy fires a
        second inference. Both the public ``on_user_turn_stopped`` event
        and the conversation context should reflect the full user
        utterance, not just the segment from the last inference.
        """
        from pipecat.frames.frames import UserTurnInferenceCompletedFrame
        from pipecat.turns.user_stop import LLMTurnCompletionUserTurnStopStrategy, deferred

        gating = LLMTurnCompletionUserTurnStopStrategy()
        upstream = deferred(
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        )
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(stop=[upstream, gating]),
            ),
        )

        inference_count = 0
        stop_message = None

        @user_aggregator.event_handler("on_user_turn_inference_triggered")
        async def on_inference_triggered(aggregator, strategy):
            nonlocal inference_count
            inference_count += 1

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_stopped(aggregator, strategy, message):
            nonlocal stop_message
            stop_message = message

        pipeline = Pipeline([user_aggregator])

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="I'm thinking", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
            # First inference fired here. Imagine the LLM returned ○;
            # the turn is not yet finalized, so the user keeps talking.
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="about pizza", user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.1),
            # Second inference fired here. Now the LLM returns ✓ and the
            # turn finalizes via UserTurnInferenceCompletedFrame.
            UserTurnInferenceCompletedFrame(),
            SleepFrame(),
        ]
        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertEqual(inference_count, 2)
        self.assertIsNotNone(stop_message)
        # The public event should report the full transcript, even
        # though each inference push only writes its own segment to
        # the context.
        self.assertEqual(stop_message.content, "I'm thinking about pizza")

        user_messages = [m for m in context.get_messages() if m.get("role") == "user"]
        self.assertEqual([m["content"] for m in user_messages], ["I'm thinking", "about pizza"])


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
        self.assertTrue(should_stop)
        self.assertIsNotNone(stop_message)
        self.assertFalse(stop_message.interrupted)
        self.assertEqual(stop_message.content, "")

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
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertFalse(stop_message.interrupted)
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
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertFalse(stop_message.interrupted)
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
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
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
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
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
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
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
            LLMContextAssistantTurnFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
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
            LLMContextAssistantTurnFrame,
            InterruptionFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
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
            LLMContextAssistantTurnFrame,
            InterruptionFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(should_start, 2)
        self.assertEqual(should_stop, 2)
        self.assertTrue(stop_messages[0].interrupted)
        self.assertEqual(stop_messages[0].content, "Hello")
        self.assertFalse(stop_messages[1].interrupted)
        self.assertEqual(stop_messages[1].content, "Hello there!")

    async def test_function_call(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)
        frames_to_send = [
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={"location": "Los Angeles"},
                cancel_on_interruption=True,
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
                cancel_on_interruption=True,
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

    async def test_tts_speak_fires_turn_started_and_stopped(self):
        """A TTSSpeakFrame(append_to_context=True) utterance must fire both turn events.

        Mirrors the TTSSpeakFrame greeting flow with a word-timestamp TTS service:
        a TTSStartedFrame opens the turn (no surrounding LLMFullResponseStartFrame),
        word-level TTSTextFrames accumulate, then the TTS service emits
        LLMAssistantPushAggregationFrame to commit them and stop the turn.
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        start_count = 0
        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal start_count
            start_count += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        frames_to_send = [
            TTSStartedFrame(append_to_context=True),
            TTSTextFrame("Hello,", aggregated_by=AggregationType.WORD),
            TTSTextFrame("how", aggregated_by=AggregationType.WORD),
            TTSTextFrame("can I help?", aggregated_by=AggregationType.WORD),
            LLMAssistantPushAggregationFrame(),
        ]
        expected_down_frames = [
            TTSStartedFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(start_count, 1)
        self.assertEqual(len(stop_messages), 1)
        self.assertFalse(stop_messages[0].interrupted)
        self.assertEqual(stop_messages[0].content, "Hello, how can I help?")
        self.assertEqual(
            context.messages[-1],
            {"role": "assistant", "content": "Hello, how can I help?"},
        )

    async def test_tts_speak_interruption_records_partial_text(self):
        """Interrupting a TTSSpeakFrame utterance records the partially-spoken text.

        The TTSStartedFrame opens the turn, so a mid-utterance interruption finds an
        open turn: the words spoken so far are written to context and
        on_assistant_turn_stopped fires with interrupted=True. (No
        LLMAssistantPushAggregationFrame arrives — the interruption cancels the
        utterance before the TTS service commits it.)
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        start_count = 0
        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal start_count
            start_count += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        frames_to_send = [
            TTSStartedFrame(append_to_context=True),
            TTSTextFrame("Let me", aggregated_by=AggregationType.WORD),
            TTSTextFrame("check on", aggregated_by=AggregationType.WORD),
            SleepFrame(),
            InterruptionFrame(),
        ]
        expected_down_frames = [
            TTSStartedFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
            InterruptionFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(start_count, 1)
        self.assertEqual(len(stop_messages), 1)
        self.assertTrue(stop_messages[0].interrupted)
        self.assertEqual(stop_messages[0].content, "Let me check on")
        self.assertEqual(
            context.messages[-1],
            {"role": "assistant", "content": "Let me check on"},
        )

    async def test_tts_started_append_to_context_false_does_not_open_turn(self):
        """A TTSStartedFrame(append_to_context=False) utterance must not open a turn.

        When the spoken text won't be written to context, neither
        on_assistant_turn_started nor on_assistant_turn_stopped should fire, and a
        following interruption must find no open turn.
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        start_count = 0
        stop_count = 0

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal start_count
            start_count += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            nonlocal stop_count
            stop_count += 1

        frames_to_send = [
            TTSStartedFrame(append_to_context=False),
            TTSTextFrame("off the record", aggregated_by=AggregationType.WORD),
            SleepFrame(),
            InterruptionFrame(),
        ]
        expected_down_frames = [TTSStartedFrame, InterruptionFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(start_count, 0)
        self.assertEqual(stop_count, 0)
        self.assertEqual(len(context.messages), 0)

    async def test_tts_started_does_not_double_open_during_llm_response(self):
        """A TTSStartedFrame during an LLM response must not re-open the turn.

        LLMFullResponseStartFrame is the earlier signal and already opened the turn,
        so the response's TTSStartedFrame is a no-op: on_assistant_turn_started fires
        exactly once.
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        start_count = 0
        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal start_count
            start_count += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSStartedFrame(append_to_context=True),
            TTSTextFrame("Hello!", aggregated_by=AggregationType.WORD),
            LLMFullResponseEndFrame(),
        ]
        await run_test(aggregator, frames_to_send=frames_to_send)
        self.assertEqual(start_count, 1)
        self.assertEqual(len(stop_messages), 1)
        self.assertEqual(stop_messages[0].content, "Hello!")

    async def test_push_aggregation_does_not_double_fire_in_llm_response(self):
        """LLMAssistantPushAggregationFrame mid-response must not double-fire turn events.

        Inside an LLMFullResponseStart/End cycle, a stray LLMAssistantPushAggregationFrame
        should flush whatever is buffered and consume the active turn (firing exactly
        one stopped event). The closing LLMFullResponseEndFrame then has no pending
        turn to stop.
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        start_count = 0
        stop_messages = []

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            nonlocal start_count
            start_count += 1

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            stop_messages.append(message)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello!"),
            LLMAssistantPushAggregationFrame(),
            LLMFullResponseEndFrame(),
        ]
        await run_test(aggregator, frames_to_send=frames_to_send)
        self.assertEqual(start_count, 1)
        self.assertEqual(len(stop_messages), 1)
        self.assertEqual(stop_messages[0].content, "Hello!")

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
            expected_up_frames=[],  # no LLMContextFrame expected, run_llm defaults to False
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
            expected_up_frames=[],  # no LLMContextFrame expected, run_llm defaults to False
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
            expected_up_frames=[],  # no LLMContextFrame expected, run_llm defaults to False
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


def _function_schema(name: str) -> FunctionSchema:
    return FunctionSchema(name=name, description="", properties={}, required=[])


def _tools(*names: str) -> ToolsSchema:
    return ToolsSchema(standard_tools=[_function_schema(n) for n in names])


def _developer_messages(context: LLMContext) -> list[str]:
    return [
        m["content"]
        for m in context.messages
        if isinstance(m, dict) and m.get("role") == "developer"
    ]


class TestToolChangeMessages(unittest.IsolatedAsyncioTestCase):
    """Coverage for the opt-in ``add_tool_change_messages`` feature.

    The feature appends a developer-role message to the context whenever
    ``LLMSetToolsFrame`` changes the set of advertised standard tools.
    """

    async def _send_set_tools_to_user_aggregator(self, aggregator, tools):
        # User aggregator forwards LLMSetToolsFrame downstream, so we expect
        # the SpeechControlParamsFrame (emitted on StartFrame) and the
        # forwarded LLMSetToolsFrame.
        await run_test(
            aggregator,
            frames_to_send=[LLMSetToolsFrame(tools=tools)],
            expected_down_frames=[SpeechControlParamsFrame, LLMSetToolsFrame],
        )

    async def test_default_off_adds_no_message(self):
        context = LLMContext(tools=_tools("a"))
        aggregator = LLMUserAggregator(context)
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("a", "b"))
        self.assertEqual(_developer_messages(context), [])

    async def test_user_aggregator_announces_additions(self):
        context = LLMContext(tools=_tools("a"))
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("a", "b", "c"))
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been added", msgs[0])
        self.assertIn("`b`", msgs[0])
        self.assertIn("`c`", msgs[0])
        self.assertNotIn("removed", msgs[0])
        # Sorted, stable order
        self.assertLess(msgs[0].index("`b`"), msgs[0].index("`c`"))

    async def test_user_aggregator_announces_removals(self):
        context = LLMContext(tools=_tools("a", "b", "c"))
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("a"))
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been removed", msgs[0])
        self.assertIn("`b`", msgs[0])
        self.assertIn("`c`", msgs[0])
        self.assertNotIn("just been added", msgs[0])

    async def test_user_aggregator_combined_add_and_remove(self):
        context = LLMContext(tools=_tools("a", "b"))
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("b", "c"))
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been added", msgs[0])
        self.assertIn("`c`", msgs[0])
        self.assertIn("just been removed", msgs[0])
        self.assertIn("`a`", msgs[0])
        # Activation phrase appears before deactivation phrase.
        self.assertLess(msgs[0].index("just been added"), msgs[0].index("just been removed"))

    async def test_no_message_when_diff_is_empty(self):
        context = LLMContext(tools=_tools("a", "b"))
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("a", "b"))
        self.assertEqual(_developer_messages(context), [])

    async def test_set_tools_to_not_given_lists_all_as_removed(self):
        from pipecat.processors.aggregators.llm_context import NOT_GIVEN

        context = LLMContext(tools=_tools("a", "b"))
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, NOT_GIVEN)
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been removed", msgs[0])
        self.assertIn("`a`", msgs[0])
        self.assertIn("`b`", msgs[0])

    async def test_set_tools_from_not_given_lists_all_as_added(self):
        context = LLMContext()  # tools default to NOT_GIVEN
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, _tools("x", "y"))
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been added", msgs[0])
        self.assertIn("`x`", msgs[0])
        self.assertIn("`y`", msgs[0])

    async def test_set_tools_accepts_plain_list(self):
        # Regression: a bare list (e.g. direct functions / FunctionSchema objects,
        # not wrapped in a ToolsSchema) must be normalized rather than raise
        # "'list' object has no attribute 'standard_tools'". Mirrors re-adding a
        # tool via LLMSetToolsFrame(tools=[...]) after it was removed.
        context = LLMContext()  # tools default to NOT_GIVEN
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        await self._send_set_tools_to_user_aggregator(aggregator, [_function_schema("b")])
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("just been added", msgs[0])
        self.assertIn("`b`", msgs[0])
        # The bare list was normalized into a ToolsSchema in the context.
        self.assertEqual({s.name for s in context.tools.standard_tools}, {"b"})

    async def test_custom_tools_only_change_no_message(self):
        # Standard tools identical; only custom tools differ → no announcement.
        context = LLMContext(
            tools=ToolsSchema(
                standard_tools=[_function_schema("a")],
                custom_tools={AdapterType.OPENAI: [{"type": "web_search"}]},
            )
        )
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(add_tool_change_messages=True)
        )
        new_tools = ToolsSchema(
            standard_tools=[_function_schema("a")],
            custom_tools={AdapterType.OPENAI: [{"type": "file_search"}]},
        )
        await self._send_set_tools_to_user_aggregator(aggregator, new_tools)
        self.assertEqual(_developer_messages(context), [])

    async def test_pipeline_with_both_aggregators_announces_once(self):
        """User agg runs first; assistant agg sees no diff and stays silent."""
        context = LLMContext(tools=_tools("a"))
        user, assistant = LLMContextAggregatorPair(context, add_tool_change_messages=True)
        pipeline = Pipeline([user, assistant])
        # The user aggregator forwards LLMSetToolsFrame downstream; the
        # assistant aggregator consumes it (does not forward).
        await run_test(
            pipeline,
            frames_to_send=[LLMSetToolsFrame(tools=_tools("a", "b"))],
            expected_down_frames=[SpeechControlParamsFrame],
        )
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1, f"expected exactly one announcement, got {msgs}")
        self.assertIn("`b`", msgs[0])

    async def test_assistant_aggregator_announces_when_handled_first(self):
        """Order-independence: an upstream LLMSetToolsFrame hits the assistant
        aggregator first (before being consumed). It should announce, and the
        user aggregator (which never sees it) shouldn't matter for correctness.
        """
        context = LLMContext(tools=_tools("a"))
        assistant = LLMAssistantAggregator(
            context,
            params=LLMAssistantAggregatorParams(add_tool_change_messages=True),
        )
        # Send the frame upstream so the assistant aggregator processes it.
        await run_test(
            assistant,
            frames_to_send=[LLMSetToolsFrame(tools=_tools("a", "b"))],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=[],
        )
        msgs = _developer_messages(context)
        self.assertEqual(len(msgs), 1)
        self.assertIn("`b`", msgs[0])

    async def test_pair_propagates_flag_to_both(self):
        context = LLMContext()
        pair = LLMContextAggregatorPair(context, add_tool_change_messages=True)
        self.assertTrue(pair.user()._add_tool_change_messages)
        self.assertTrue(pair.assistant()._add_tool_change_messages)

    async def test_pair_arg_overrides_per_params_settings(self):
        context = LLMContext()
        pair = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(add_tool_change_messages=False),
            assistant_params=LLMAssistantAggregatorParams(add_tool_change_messages=False),
            add_tool_change_messages=True,
        )
        self.assertTrue(pair.user()._add_tool_change_messages)
        self.assertTrue(pair.assistant()._add_tool_change_messages)

    async def test_pair_default_respects_per_params(self):
        context = LLMContext()
        pair = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(add_tool_change_messages=True),
            assistant_params=LLMAssistantAggregatorParams(add_tool_change_messages=False),
        )
        self.assertTrue(pair.user()._add_tool_change_messages)
        self.assertFalse(pair.assistant()._add_tool_change_messages)


class TestRealtimeServiceModeAggregator(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for the trailing-write realtime mode."""

    def _build_pair(
        self,
        *,
        realtime_service_mode: bool = False,
        user_params: LLMUserAggregatorParams | None = None,
    ) -> tuple[LLMContext, LLMContextAggregatorPair]:
        context = LLMContext()
        pair = LLMContextAggregatorPair(
            context,
            user_params=user_params,
            realtime_service_mode=realtime_service_mode,
        )
        return context, pair

    async def test_pair_propagates_realtime_mode_to_halves(self):
        _, pair = self._build_pair(realtime_service_mode=True)
        # Realtime mode is one-way: the assistant holds the back-ref so
        # it can flush the user on LLMFullResponseStartFrame. The user
        # has no back-ref — the assistant writes its own message on
        # LLMFullResponseEndFrame, so it doesn't need to call back.
        self.assertIs(pair.assistant()._paired_user_aggregator, pair.user())
        self.assertTrue(pair.user()._realtime_service_mode)
        self.assertTrue(pair.assistant()._realtime_service_mode)

    async def test_pair_omits_realtime_wiring_when_unset(self):
        _, pair = self._build_pair()
        self.assertIsNone(pair.assistant()._paired_user_aggregator)
        self.assertFalse(pair.user()._realtime_service_mode)
        self.assertFalse(pair.assistant()._realtime_service_mode)

    async def test_realtime_strategy_mutations_with_defaults(self):
        # At __init__ time only the mutations apply (drop the
        # transcription start strategy, flip wait_for_transcript on
        # default stops). The external-strategy replacement is deferred
        # to the RealtimeServiceMetadataFrame handler.
        _, pair = self._build_pair(realtime_service_mode=True)
        strategies = pair.user()._user_turn_controller._user_turn_strategies
        for s in strategies.start:
            self.assertNotIsInstance(s, TranscriptionUserTurnStartStrategy)
        # Default VAD start strategy is preserved; no External strategies
        # installed yet.
        self.assertTrue(any(isinstance(s, VADUserTurnStartStrategy) for s in strategies.start))
        self.assertFalse(
            any(isinstance(s, ExternalUserTurnStartStrategy) for s in strategies.start)
        )
        self.assertFalse(any(isinstance(s, ExternalUserTurnStopStrategy) for s in strategies.stop))
        for s in strategies.stop:
            if hasattr(s, "wait_for_transcript"):
                self.assertFalse(s.wait_for_transcript)

    async def test_realtime_metadata_replaces_defaults_when_service_emits_turn_frames(self):
        # When the service advertises emits_user_turn_frames=True and
        # the user didn't pass custom strategies, the handler swaps the
        # defaults out for ExternalUserTurnStart/StopStrategy.
        _, pair = self._build_pair(realtime_service_mode=True)
        frames_to_send = [
            RealtimeServiceMetadataFrame(
                service_name="FakeRealtimeLLM", emits_user_turn_frames=True
            ),
        ]
        await run_test(Pipeline([pair.user(), pair.assistant()]), frames_to_send=frames_to_send)
        strategies = pair.user()._user_turn_controller._user_turn_strategies
        self.assertEqual(len(strategies.start), 1)
        self.assertIsInstance(strategies.start[0], ExternalUserTurnStartStrategy)
        self.assertEqual(len(strategies.stop), 1)
        self.assertIsInstance(strategies.stop[0], ExternalUserTurnStopStrategy)
        # Realtime-mode mutation is reapplied to the new stop strategy.
        self.assertFalse(strategies.stop[0].wait_for_transcript)

    async def test_realtime_metadata_keeps_defaults_when_service_does_not_emit_turn_frames(self):
        # Services advertising emits_user_turn_frames=False keep the
        # default strategies so locally-driven turns (e.g. local VAD)
        # can fire on_user_turn_* events.
        _, pair = self._build_pair(realtime_service_mode=True)
        frames_to_send = [
            RealtimeServiceMetadataFrame(
                service_name="FakeRealtimeLLM", emits_user_turn_frames=False
            ),
        ]
        await run_test(Pipeline([pair.user(), pair.assistant()]), frames_to_send=frames_to_send)
        strategies = pair.user()._user_turn_controller._user_turn_strategies
        self.assertFalse(
            any(isinstance(s, ExternalUserTurnStartStrategy) for s in strategies.start)
        )
        self.assertFalse(any(isinstance(s, ExternalUserTurnStopStrategy) for s in strategies.stop))
        self.assertTrue(any(isinstance(s, VADUserTurnStartStrategy) for s in strategies.start))

    async def test_realtime_metadata_keeps_custom_strategies(self):
        # Custom user_turn_strategies opts out of the swap — explicit
        # user choice wins, regardless of what the service advertises.
        custom = UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy()],
        )
        _, pair = self._build_pair(
            realtime_service_mode=True,
            user_params=LLMUserAggregatorParams(user_turn_strategies=custom),
        )
        frames_to_send = [
            RealtimeServiceMetadataFrame(
                service_name="FakeRealtimeLLM", emits_user_turn_frames=True
            ),
        ]
        await run_test(Pipeline([pair.user(), pair.assistant()]), frames_to_send=frames_to_send)
        strategies = pair.user()._user_turn_controller._user_turn_strategies
        self.assertFalse(
            any(isinstance(s, ExternalUserTurnStartStrategy) for s in strategies.start)
        )
        self.assertFalse(any(isinstance(s, ExternalUserTurnStopStrategy) for s in strategies.stop))
        self.assertTrue(any(isinstance(s, VADUserTurnStartStrategy) for s in strategies.start))

    async def test_trailing_write_user_then_assistant_then_user(self):
        _, pair = self._build_pair(realtime_service_mode=True)
        user, assistant = pair

        user_msg_added: list[UserTurnMessageAddedMessage] = []
        assistant_msg_stopped: list[AssistantTurnStoppedMessage] = []

        @user.event_handler("on_user_turn_message_added")
        async def _on_um(_a, msg):
            user_msg_added.append(msg)

        @assistant.event_handler("on_assistant_turn_stopped")
        async def _on_ats(_a, msg):
            assistant_msg_stopped.append(msg)

        context = user.context

        # Sequence: user transcript, assistant response starts (flushes
        # user), assistant response ends (writes assistant), new user
        # transcript, EndFrame flushes the new user message.
        frames_to_send = [
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now"),
            SleepFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hi "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
            SleepFrame(),
            TranscriptionFrame(text="How are you?", user_id="", timestamp="now"),
            SleepFrame(),
        ]
        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=frames_to_send,
        )

        # Context should contain: user("Hello!"), assistant("Hi there!"),
        # user("How are you?").
        messages = context.get_messages()
        roles_contents = [(m["role"], m["content"]) for m in messages]
        self.assertEqual(
            roles_contents,
            [
                ("user", "Hello!"),
                ("assistant", "Hi there!"),
                ("user", "How are you?"),
            ],
        )
        self.assertEqual([m.content for m in user_msg_added], ["Hello!", "How are you?"])
        self.assertEqual([m.content for m in assistant_msg_stopped], ["Hi there!"])
        for msg in assistant_msg_stopped:
            self.assertFalse(msg.interrupted)

    async def test_interruption_writes_assistant_immediately(self):
        _, pair = self._build_pair(realtime_service_mode=True)
        user, assistant = pair

        assistant_messages: list[AssistantTurnStoppedMessage] = []

        @assistant.event_handler("on_assistant_turn_stopped")
        async def _on_ats(_a, msg):
            assistant_messages.append(msg)

        context = user.context

        frames_to_send = [
            TranscriptionFrame(text="Hi!", user_id="", timestamp="now"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
        ]
        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=frames_to_send,
        )

        roles_contents = [(m["role"], m["content"]) for m in context.get_messages()]
        # User message written when assistant started; assistant message
        # written immediately on interruption with interrupted=True.
        self.assertEqual(roles_contents, [("user", "Hi!"), ("assistant", "Hello")])
        self.assertEqual(len(assistant_messages), 1)
        self.assertTrue(assistant_messages[0].interrupted)

    async def test_user_turn_stopped_in_realtime_mode_has_none_content(self):
        # When VAD turn frames fire in realtime mode, the user-turn-stop
        # message carries content=None — the message isn't finalized yet.
        _, pair = self._build_pair(
            realtime_service_mode=True,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(
                            user_speech_timeout=TRANSCRIPTION_TIMEOUT,
                        )
                    ],
                ),
                user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
            ),
        )
        user, assistant = pair

        stop_messages: list[UserTurnStoppedMessage] = []

        @user.event_handler("on_user_turn_stopped")
        async def _on_stop(_a, _s, msg):
            stop_messages.append(msg)

        frames_to_send = [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text="hey", user_id="", timestamp="now"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=TRANSCRIPTION_TIMEOUT + 0.05),
        ]
        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=frames_to_send,
        )
        self.assertEqual(len(stop_messages), 1)
        self.assertIsNone(stop_messages[0].content)

    async def test_realtime_metadata_recommendation_log_when_unconfigured(self):
        # Cascade pair receiving a RealtimeServiceMetadataFrame logs the
        # one-time recommendation. The user half records the fact via
        # _realtime_recommendation_logged.
        _, pair = self._build_pair()
        user = pair.user()

        frames_to_send = [
            RealtimeServiceMetadataFrame(
                service_name="FakeRealtimeLLM", emits_user_turn_frames=False
            ),
        ]
        await run_test(Pipeline([pair.user(), pair.assistant()]), frames_to_send=frames_to_send)
        self.assertTrue(user._realtime_recommendation_logged)

    async def test_realtime_metadata_no_log_when_configured(self):
        # When realtime mode is opted in, the metadata frame is consumed
        # without firing the recommendation log (we still flag the
        # one-shot bookkeeping).
        _, pair = self._build_pair(realtime_service_mode=True)
        user = pair.user()

        frames_to_send = [
            RealtimeServiceMetadataFrame(
                service_name="FakeRealtimeLLM", emits_user_turn_frames=False
            ),
        ]
        await run_test(Pipeline([pair.user(), pair.assistant()]), frames_to_send=frames_to_send)
        self.assertTrue(user._realtime_recommendation_logged)

    async def test_realtime_mode_assistant_requires_paired_user_aggregator(self):
        # Direct construction of the assistant half with realtime mode
        # set but no paired user half raises at StartFrame validation.
        # (We call the validation directly so the error isn't swallowed
        # by the pipeline's exception handler.)
        context = LLMContext()
        assistant = LLMAssistantAggregator(context, _realtime_service_mode=True)
        with self.assertRaises(RuntimeError):
            assistant._validate_realtime_pairing()

    async def test_realtime_mode_assistant_rejects_mismatched_halves(self):
        # If a user code path constructs halves with mismatched configs
        # and wires them up by hand, assistant validation catches it.
        context = LLMContext()
        user = LLMUserAggregator(context, _realtime_service_mode=True)
        assistant = LLMAssistantAggregator(
            context,
            _realtime_service_mode=False,
            _paired_user_aggregator=user,
        )
        with self.assertRaises(RuntimeError):
            assistant._validate_realtime_pairing()


if __name__ == "__main__":
    unittest.main()
