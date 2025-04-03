#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from typing import Any

import google.ai.generativelanguage as glm

from pipecat.frames.frames import (
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OpenAILLMContextAssistantTimestampFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.anthropic.llm import (
    AnthropicAssistantContextAggregator,
    AnthropicLLMContext,
    AnthropicUserContextAggregator,
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
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
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
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
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
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
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
        aggregator = self.AGGREGATOR_CLASS(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            InterimTranscriptionFrame(text="Hello ", user_id="cat", timestamp=""),
            SleepFrame(),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [*self.EXPECTED_CONTEXT_FRAMES]
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
            context,
            aggregation_timeout=AGGREGATION_TIMEOUT,
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


class BaseTestAssistantContextAggreagator:
    CONTEXT_CLASS = None  # To be set in subclasses
    AGGREGATOR_CLASS = None  # To be set in subclasses
    EXPECTED_CONTEXT_FRAMES = None  # To be set in subclasses

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
        aggregator = self.AGGREGATOR_CLASS(context, expect_stripped_words=False)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello "),
            TextFrame(text="Pipecat. "),
            TextFrame(text="How are "),
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
        aggregator = self.AGGREGATOR_CLASS(context, expect_stripped_words=False)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello "),
            TextFrame(text="Pipecat."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            TextFrame(text="How are "),
            TextFrame(text="you?"),
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
        aggregator = self.AGGREGATOR_CLASS(context, expect_stripped_words=False)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello "),
            TextFrame(text="Pipecat."),
            LLMFullResponseEndFrame(),
            SleepFrame(AGGREGATION_SLEEP),
            StartInterruptionFrame(),
            LLMFullResponseStartFrame(),
            TextFrame(text="How are "),
            TextFrame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            *self.EXPECTED_CONTEXT_FRAMES,
            StartInterruptionFrame,
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
# OpenAI
#


class TestOpenAIUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = OpenAIUserContextAggregator


class TestOpenAIAssistantContextAggregator(
    BaseTestAssistantContextAggreagator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = OpenAIAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]


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
    BaseTestAssistantContextAggreagator, unittest.IsolatedAsyncioTestCase
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
# Google
#


class TestGoogleUserContextAggregator(
    BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = GoogleLLMContext
    AGGREGATOR_CLASS = GoogleUserContextAggregator

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        obj = glm.Content.to_dict(context.messages[index])
        assert obj["parts"][0]["text"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        obj = glm.Content.to_dict(context.messages[index])
        assert obj["parts"][0]["text"] == content


class TestGoogleAssistantContextAggregator(
    BaseTestAssistantContextAggreagator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = GoogleLLMContext
    AGGREGATOR_CLASS = GoogleAssistantContextAggregator
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame, OpenAILLMContextAssistantTimestampFrame]

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        obj = glm.Content.to_dict(context.messages[index])
        assert obj["parts"][0]["text"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        obj = glm.Content.to_dict(context.messages[index])
        assert obj["parts"][0]["text"] == content

    def check_function_call_result(self, context: OpenAILLMContext, index: int, content: Any):
        obj = glm.Content.to_dict(context.messages[index])
        assert obj["parts"][0]["function_response"]["response"]["value"] == json.dumps(content)
