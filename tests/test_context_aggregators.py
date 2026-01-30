#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from typing import Any, Optional

from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterimTranscriptionFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OpenAILLMContextAssistantTimestampFrame,
    SpeechControlParamsFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import (
    AnthropicAssistantContextAggregator,
    AnthropicLLMContext,
    AnthropicUserContextAggregator,
)
from pipecat.services.aws.llm import (
    AWSBedrockAssistantContextAggregator,
    AWSBedrockLLMContext,
    AWSBedrockUserContextAggregator,
)
from pipecat.services.google.llm import (
    GoogleAssistantContextAggregator,
    GoogleLLMContext,
    GoogleUserContextAggregator,
)
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.tests.utils import SleepFrame, run_test

AGGREGATION_TIMEOUT = 0.1
AGGREGATION_SLEEP = 0.15


class BaseTestUserContextAggregator:
    CONTEXT_CLASS = None  # To be set in subclasses
    AGGREGATOR_CLASS = None  # To be set in subclasses
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame]

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        assert context.messages[index]["content"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        assert context.messages[index]["content"] == content

    async def test_se(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_ste(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            UserStoppedSpeakingFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello!")

    async def test_site(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp=""),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            UserStoppedSpeakingFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat!")

    async def test_st1iest2e(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            UserStoppedSpeakingFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat! How are you?")

    async def test_siet(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "How are you?")

    async def test_sieit(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "How are you?")

    async def test_set(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "How are you?")

    async def test_seit(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "How are you?")

    async def test_st1et2(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            SpeechControlParamsFrame(vad_params=VADParams(stop_secs=AGGREGATION_TIMEOUT)),
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            SpeechControlParamsFrame,
            UserStartedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_multi_content(context, 0, 0, "Hello Pipecat!")
        self.check_message_multi_content(context, 0, 1, "How are you?")

    async def test_set1t2(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat! How are you?")

    async def test_siet1it2(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="Hello ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat! How are you?")

    async def test_t(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context
        )  # No aggregation timeout; this tests VAD emulation

        frames_to_send = [
            SpeechControlParamsFrame(vad_params=VADParams(stop_secs=AGGREGATION_TIMEOUT)),
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            SpeechControlParamsFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        expected_up_frames = [EmulateUserStartedSpeakingFrame, EmulateUserStoppedSpeakingFrame]

        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        self.check_message_content(context, 0, "Hello!")

    async def test_t_with_turn_analyzer(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(turn_emulated_vad_timeout=AGGREGATION_TIMEOUT)
        )

        frames_to_send = [
            SpeechControlParamsFrame(
                vad_params=VADParams(stop_secs=0.2),
                turn_params=SmartTurnParams(stop_secs=3.0),  # Turn analyzer present
            ),
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            SpeechControlParamsFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        expected_up_frames = [EmulateUserStartedSpeakingFrame, EmulateUserStoppedSpeakingFrame]

        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        self.check_message_content(context, 0, "Hello!")

    async def test_it(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context
        )  # No aggregation timeout; this tests VAD emulation
        frames_to_send = [
            SpeechControlParamsFrame(vad_params=VADParams(stop_secs=AGGREGATION_TIMEOUT)),
            InterimTranscriptionFrame(text="Hello ", user_id="cat", timestamp=""),
            SleepFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [SpeechControlParamsFrame, *self.EXPECTED_CONTEXT_FRAMES]
        expected_up_frames = [EmulateUserStartedSpeakingFrame, EmulateUserStoppedSpeakingFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat!")

    async def test_sie_delay_it(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=LLMUserAggregatorParams(aggregation_timeout=AGGREGATION_TIMEOUT)
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(AGGREGATION_SLEEP),
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "How are you?")

    async def test_min_words_interruption_strategy_one_word(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        class ContextProcessor(FrameProcessor):
            def __init__(self):
                super().__init__()
                self.context_received = False

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, OpenAILLMContextFrame):
                    self.context_received = True

                await self.push_frame(frame, direction)

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        context_processor = ContextProcessor()
        pipeline = Pipeline([aggregator, context_processor])

        frames_to_send = [
            BotStartedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Can", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            pipeline_params=PipelineParams(
                interruption_strategies=[MinWordsInterruptionStrategy(min_words=2)]
            ),
        )
        assert not context_processor.context_received

    async def test_min_words_interruption_strategy_two_words(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        class ContextProcessor(FrameProcessor):
            def __init__(self):
                super().__init__()
                self.context_received = False

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, OpenAILLMContextFrame):
                    self.context_received = True
                elif isinstance(frame, InterruptionFrame):
                    self.context_received = False

                await self.push_frame(frame, direction)

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        context_processor = ContextProcessor()
        pipeline = Pipeline([aggregator, context_processor])

        frames_to_send = [
            BotStartedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Can you", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_up_frames = [InterruptionTaskFrame]
        expected_down_frames = [
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InterruptionFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_up_frames=expected_up_frames,
            expected_down_frames=expected_down_frames,
            pipeline_params=PipelineParams(
                interruption_strategies=[MinWordsInterruptionStrategy(min_words=2)]
            ),
        )
        self.check_message_content(context, 0, "Can you")
        # If the context is not received or it has been cleared by the
        # interruption then we have an issue.
        assert context_processor.context_received


class BaseTestAssistantContextAggregator:
    CONTEXT_CLASS = None  # To be set in subclasses
    AGGREGATOR_CLASS = None  # To be set in subclasses
    EXPECTED_CONTEXT_FRAMES = None  # To be set in subclasses

    def create_assistant_aggregator_params(
        self, **kwargs
    ) -> Optional[LLMAssistantAggregatorParams]:
        return LLMAssistantAggregatorParams(**kwargs)

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        assert context.messages[index]["content"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        assert context.messages[index]["content"] == content

    def check_function_call_result(self, context: OpenAILLMContext, index: int, content: str):
        assert json.loads(context.messages[index]["content"]) == content

    async def test_empty(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]
        expected_down_frames = []
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_single_text(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat!")

    async def test_multiple_text(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=self.create_assistant_aggregator_params(expect_stripped_words=False)
        )

        # The newer LLMAssistantAggregator expects TextFrames to declare
        # when they include inter-frame spaces.
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
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat. How are you?")

    async def test_multiple_text_stripped(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello"),
            TextFrame(text="Pipecat."),
            TextFrame(text="How are"),
            TextFrame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(context, 0, "Hello Pipecat. How are you?")

    async def test_multiple_llm_responses(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=self.create_assistant_aggregator_params(expect_stripped_words=False)
        )

        # The newer LLMAssistantAggregator expects TextFrames to declare
        # when they include inter-frame spaces.
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
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES, *self.EXPECTED_CONTEXT_FRAMES]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_multi_content(context, 0, 0, "Hello Pipecat.")
        self.check_message_multi_content(context, 0, 1, "How are you?")

    async def test_multiple_llm_responses_interruption(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=self.create_assistant_aggregator_params(expect_stripped_words=False)
        )

        # The newer LLMAssistantAggregator expects TextFrames to declare
        # when they include inter-frame spaces.
        def make_text_frame(text: str) -> TextFrame:
            frame = TextFrame(text=text)
            frame.includes_inter_frame_spaces = True
            return frame

        frames_to_send = [
            LLMFullResponseStartFrame(),
            make_text_frame("Hello "),
            make_text_frame("Pipecat."),
            LLMFullResponseEndFrame(),
            SleepFrame(AGGREGATION_SLEEP),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            make_text_frame("How are "),
            make_text_frame("you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            *self.EXPECTED_CONTEXT_FRAMES,
            InterruptionFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_multi_content(context, 0, 0, "Hello Pipecat.")
        self.check_message_multi_content(context, 0, 1, "How are you?")

    async def test_function_call(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
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
        self.check_function_call_result(context, -1, {"conditions": "Sunny"})

    async def test_function_call_on_context_updated(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context_updated = False

        async def on_context_updated():
            nonlocal context_updated
            context_updated = True

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
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
        self.check_function_call_result(context, -1, {"conditions": "Sunny"})
        assert context_updated


#
# LLMUserContextAggregator
#


class TestLLMUserContextAggregator(BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = LLMUserContextAggregator


#
# Anthropic
#


class TestAnthropicUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = AnthropicLLMContext
    AGGREGATOR_CLASS = AnthropicUserContextAggregator

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        messages = context.messages[content_index]
        assert messages["content"][index]["text"] == content


class TestAnthropicAssistantContextAggregator(
    BaseTestAssistantContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = AnthropicLLMContext
    AGGREGATOR_CLASS = AnthropicAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        messages = context.messages[content_index]
        assert messages["content"][index]["text"] == content

    def check_function_call_result(self, context: OpenAILLMContext, index: int, content: Any):
        assert context.messages[index]["content"][0]["content"] == json.dumps(content)


#
# AWS (Bedrock)
#


class TestAWSBedrockUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = AWSBedrockLLMContext
    AGGREGATOR_CLASS = AWSBedrockUserContextAggregator

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        messages = context.messages[content_index]
        assert messages["content"][index]["text"] == content


class TestAWSBedrockAssistantContextAggregator(
    BaseTestAssistantContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = AWSBedrockLLMContext
    AGGREGATOR_CLASS = AWSBedrockAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        messages = context.messages[content_index]
        assert messages["content"][index]["text"] == content

    def check_function_call_result(self, context: OpenAILLMContext, index: int, content: Any):
        assert context.messages[index]["content"][0]["toolResult"]["content"][0][
            "text"
        ] == json.dumps(content)


#
# Google
#


class TestGoogleUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = GoogleLLMContext
    AGGREGATOR_CLASS = GoogleUserContextAggregator

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        obj = context.messages[index].to_json_dict()
        assert obj["parts"][0]["text"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        obj = context.messages[index].to_json_dict()
        assert obj["parts"][0]["text"] == content


class TestGoogleAssistantContextAggregator(
    BaseTestAssistantContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = GoogleLLMContext
    AGGREGATOR_CLASS = GoogleAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        obj = context.messages[index].to_json_dict()
        assert obj["parts"][0]["text"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        obj = context.messages[index].to_json_dict()
        assert obj["parts"][0]["text"] == content

    def check_function_call_result(self, context: OpenAILLMContext, index: int, content: Any):
        obj = context.messages[index].to_json_dict()
        assert obj["parts"][0]["function_response"]["response"]["value"] == json.dumps(content)


#
# OpenAI
#


class TestOpenAIUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = OpenAIUserContextAggregator


class TestOpenAIAssistantContextAggregator(
    BaseTestAssistantContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = OpenAIAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]


#
# Universal
#
class TestLLMAssistantAggregator(
    BaseTestAssistantContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = LLMContext
    AGGREGATOR_CLASS = LLMAssistantAggregator
    EXPECTED_CONTEXT_FRAMES = [LLMContextFrame, LLMContextAssistantTimestampFrame]

    # Override to remove 'expect_stripped_words' parameter, which is deprecated
    # for LLMAssistantAggregator
    def create_assistant_aggregator_params(
        self, **kwargs
    ) -> Optional[LLMAssistantAggregatorParams]:
        kwargs.pop("expect_stripped_words", None)
        return LLMAssistantAggregatorParams(**kwargs) if kwargs else None

    async def test_multiple_text_mixed(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(
            context, params=self.create_assistant_aggregator_params(expect_stripped_words=False)
        )

        # The newer LLMAssistantAggregator expects TextFrames to declare
        # when they include inter-frame spaces.
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
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_content(
            context,
            0,
            "Hello Pipecat. Here's some code: ```python\nprint('Hello, World!')\n``` ```javascript\nconsole.log('Hello, World!');\n``` And some more: ```html\n<div>Hello, World!</div>\n``` Hope that helps!",
        )


class TestLLMContextDiff(unittest.TestCase):
    """Tests for the LLMContext.diff() method."""

    def test_diff_identical_contexts(self):
        """Test diff of two identical contexts returns no changes."""
        messages = [{"role": "user", "content": "Hello"}]
        context1 = LLMContext(messages=messages.copy())
        context2 = LLMContext(messages=messages.copy())

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())
        self.assertEqual(diff.messages_appended, [])
        self.assertFalse(diff.history_edited)
        self.assertEqual(diff.tool_calls_resolved, [])
        self.assertFalse(diff.tools_diff.has_changes())
        self.assertFalse(diff.tool_choice_changed)

    def test_diff_messages_appended(self):
        """Test diff detects appended messages."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "assistant", "content": "Hi there!"}

        context1 = LLMContext(messages=[msg1])
        context2 = LLMContext(messages=[msg1, msg2])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(len(diff.messages_appended), 1)
        self.assertEqual(diff.messages_appended[0], msg2)
        self.assertFalse(diff.history_edited)

    def test_diff_multiple_messages_appended(self):
        """Test diff detects multiple appended messages."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "assistant", "content": "Hi!"}
        msg3 = {"role": "user", "content": "How are you?"}

        context1 = LLMContext(messages=[msg1])
        context2 = LLMContext(messages=[msg1, msg2, msg3])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(len(diff.messages_appended), 2)
        self.assertEqual(diff.messages_appended[0], msg2)
        self.assertEqual(diff.messages_appended[1], msg3)
        self.assertFalse(diff.history_edited)

    def test_diff_message_removed(self):
        """Test diff detects message removal as history edit."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "assistant", "content": "Hi!"}

        context1 = LLMContext(messages=[msg1, msg2])
        context2 = LLMContext(messages=[msg1])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.messages_appended, [])  # Empty when history edited
        self.assertTrue(diff.history_edited)

    def test_diff_message_modified(self):
        """Test diff detects message modification as history edit."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2_v1 = {"role": "assistant", "content": "Hi!"}
        msg2_v2 = {"role": "assistant", "content": "Hello there!"}

        context1 = LLMContext(messages=[msg1, msg2_v1])
        context2 = LLMContext(messages=[msg1, msg2_v2])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.history_edited)
        self.assertEqual(diff.messages_appended, [])

    def test_diff_message_inserted_in_middle(self):
        """Test diff detects message insertion in middle as history edit."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "assistant", "content": "Hi!"}
        msg_inserted = {"role": "system", "content": "System message"}

        context1 = LLMContext(messages=[msg1, msg2])
        context2 = LLMContext(messages=[msg1, msg_inserted, msg2])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.history_edited)
        self.assertEqual(diff.messages_appended, [])

    def test_diff_tool_call_resolved_to_result(self):
        """Test diff detects tool call resolution to actual result."""
        msg1 = {"role": "user", "content": "What's the weather?"}
        msg2 = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        }
        tool_in_progress = {"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_123"}
        tool_resolved = {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_123",
        }

        context1 = LLMContext(messages=[msg1, msg2, tool_in_progress])
        context2 = LLMContext(messages=[msg1, msg2, tool_resolved])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.tool_calls_resolved, ["call_123"])
        # Note: the tool message content changed, so history is edited
        self.assertTrue(diff.history_edited)

    def test_diff_tool_call_resolved_to_completed(self):
        """Test diff detects tool call resolution to COMPLETED."""
        msg1 = {"role": "user", "content": "Do something"}
        tool_in_progress = {"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_456"}
        tool_completed = {"role": "tool", "content": "COMPLETED", "tool_call_id": "call_456"}

        context1 = LLMContext(messages=[msg1, tool_in_progress])
        context2 = LLMContext(messages=[msg1, tool_completed])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.tool_calls_resolved, ["call_456"])

    def test_diff_tool_call_resolved_to_cancelled(self):
        """Test diff detects tool call resolution to CANCELLED."""
        msg1 = {"role": "user", "content": "Do something"}
        tool_in_progress = {"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_789"}
        tool_cancelled = {"role": "tool", "content": "CANCELLED", "tool_call_id": "call_789"}

        context1 = LLMContext(messages=[msg1, tool_in_progress])
        context2 = LLMContext(messages=[msg1, tool_cancelled])

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.tool_calls_resolved, ["call_789"])

    def test_diff_tool_call_still_in_progress(self):
        """Test diff does not report tool call as resolved if still IN_PROGRESS."""
        msg1 = {"role": "user", "content": "Do something"}
        tool_in_progress = {"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_123"}

        context1 = LLMContext(messages=[msg1, tool_in_progress])
        context2 = LLMContext(messages=[msg1, tool_in_progress])

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())
        self.assertEqual(diff.tool_calls_resolved, [])

    def test_diff_tool_choice_changed(self):
        """Test diff detects tool_choice changes."""
        msg1 = {"role": "user", "content": "Hello"}

        context1 = LLMContext(messages=[msg1], tool_choice="auto")
        context2 = LLMContext(messages=[msg1], tool_choice="none")

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.tool_choice_changed)

    def test_diff_tool_choice_unchanged(self):
        """Test diff reports no change when tool_choice is the same."""
        msg1 = {"role": "user", "content": "Hello"}

        context1 = LLMContext(messages=[msg1], tool_choice="auto")
        context2 = LLMContext(messages=[msg1], tool_choice="auto")

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())
        self.assertFalse(diff.tool_choice_changed)

    def test_diff_empty_contexts(self):
        """Test diff of two empty contexts returns no changes."""
        context1 = LLMContext()
        context2 = LLMContext()

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())


class TestLLMContextDiffWithTools(unittest.TestCase):
    """Tests for LLMContext.diff() with tools configuration changes."""

    def _create_tools_schema(self, tool_names: list[str]) -> "ToolsSchema":
        """Helper to create a ToolsSchema with named tools."""
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = [
            FunctionSchema(name=name, description=f"Test {name}", properties={}, required=[])
            for name in tool_names
        ]
        return ToolsSchema(standard_tools=tools)

    def test_diff_tools_added_from_not_given(self):
        """Test diff detects tools being added when self has no tools."""
        from pipecat.processors.aggregators.llm_context import NOT_GIVEN

        msg1 = {"role": "user", "content": "Hello"}
        tools = self._create_tools_schema(["get_weather", "get_time"])

        context1 = LLMContext(messages=[msg1], tools=NOT_GIVEN)
        context2 = LLMContext(messages=[msg1], tools=tools)

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(sorted(diff.tools_diff.standard_tools_added), ["get_time", "get_weather"])
        self.assertEqual(diff.tools_diff.standard_tools_removed, [])

    def test_diff_tools_removed_to_not_given(self):
        """Test diff detects tools being removed when other has no tools."""
        from pipecat.processors.aggregators.llm_context import NOT_GIVEN

        msg1 = {"role": "user", "content": "Hello"}
        tools = self._create_tools_schema(["get_weather", "get_time"])

        context1 = LLMContext(messages=[msg1], tools=tools)
        context2 = LLMContext(messages=[msg1], tools=NOT_GIVEN)

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertEqual(diff.tools_diff.standard_tools_added, [])
        self.assertEqual(
            sorted(diff.tools_diff.standard_tools_removed), ["get_time", "get_weather"]
        )

    def test_diff_both_not_given(self):
        """Test diff returns None tools_diff when both have no tools."""
        from pipecat.processors.aggregators.llm_context import NOT_GIVEN

        msg1 = {"role": "user", "content": "Hello"}

        context1 = LLMContext(messages=[msg1], tools=NOT_GIVEN)
        context2 = LLMContext(messages=[msg1], tools=NOT_GIVEN)

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())
        self.assertFalse(diff.tools_diff.has_changes())

    def test_diff_tools_modified(self):
        """Test diff detects tool modification via ToolsSchema.diff()."""
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        msg1 = {"role": "user", "content": "Hello"}

        tool_v1 = FunctionSchema(
            name="get_weather",
            description="Get weather v1",
            properties={"location": {"type": "string"}},
            required=["location"],
        )
        tool_v2 = FunctionSchema(
            name="get_weather",
            description="Get weather v2",
            properties={"city": {"type": "string"}},
            required=["city"],
        )

        context1 = LLMContext(messages=[msg1], tools=ToolsSchema(standard_tools=[tool_v1]))
        context2 = LLMContext(messages=[msg1], tools=ToolsSchema(standard_tools=[tool_v2]))

        diff = context1.diff(context2)
        self.assertTrue(diff.has_changes())
        self.assertTrue(diff.tools_diff.standard_tools_modified)

    def test_diff_tools_unchanged(self):
        """Test diff returns None tools_diff when tools are identical."""
        msg1 = {"role": "user", "content": "Hello"}
        tools1 = self._create_tools_schema(["get_weather"])
        tools2 = self._create_tools_schema(["get_weather"])

        context1 = LLMContext(messages=[msg1], tools=tools1)
        context2 = LLMContext(messages=[msg1], tools=tools2)

        diff = context1.diff(context2)
        self.assertFalse(diff.has_changes())
        self.assertFalse(diff.tools_diff.has_changes())
