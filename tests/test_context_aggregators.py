#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import google.ai.generativelanguage as glm

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    OpenAILLMContextAssistantTimestampFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.anthropic import (
    AnthropicAssistantContextAggregator,
    AnthropicLLMContext,
    AnthropicUserContextAggregator,
)
from pipecat.services.google.google import (
    GoogleAssistantContextAggregator,
    GoogleLLMContext,
    GoogleUserContextAggregator,
)
from pipecat.services.openai import OpenAIAssistantContextAggregator, OpenAIUserContextAggregator
from pipecat.tests.utils import SleepFrame, run_test

AGGREGATION_TIMEOUT = 0.1
AGGREGATION_SLEEP = 0.15
BOT_INTERRUPTION_TIMEOUT = 0.2
BOT_INTERRUPTION_SLEEP = 0.25


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
            bot_interruption_timeout=BOT_INTERRUPTION_TIMEOUT,
        )
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(BOT_INTERRUPTION_SLEEP),
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        expected_up_frames = [EmulateUserStartedSpeakingFrame, EmulateUserStoppedSpeakingFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        self.check_message_content(context, 0, "How are you?")


class BaseTestAssistantContextAggreagator:
    CONTEXT_CLASS = None  # To be set in subclasses
    AGGREGATOR_CLASS = None  # To be set in subclasses
    EXPECTED_CONTEXT_FRAMES = [OpenAILLMContextFrame]

    def check_message_content(self, context: OpenAILLMContext, index: int, content: str):
        assert context.messages[index]["content"] == content

    def check_message_multi_content(
        self, context: OpenAILLMContext, content_index: int, index: int, content: str
    ):
        assert context.messages[index]["content"] == content

    async def test_single_text(self):
        assert self.CONTEXT_CLASS is not None, "CONTEXT_CLASS must be set in a subclass"
        assert self.AGGREGATOR_CLASS is not None, "AGGREGATOR_CLASS must be set in a subclass"

        context = self.CONTEXT_CLASS()
        aggregator = self.AGGREGATOR_CLASS(context)
        frames_to_send = [
            TTSTextFrame(text="Hello Pipecat!"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]
        expected_down_frames = [BotStoppedSpeakingFrame, *self.EXPECTED_CONTEXT_FRAMES]
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
            TTSTextFrame(text="Hello "),
            TTSTextFrame(text="Pipecat. "),
            TTSTextFrame(text="How are "),
            TTSTextFrame(text="you?"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]
        expected_down_frames = [BotStoppedSpeakingFrame, *self.EXPECTED_CONTEXT_FRAMES]
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
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="Pipecat."),
            TTSTextFrame(text="How are"),
            TTSTextFrame(text="you?"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]
        expected_down_frames = [BotStoppedSpeakingFrame, *self.EXPECTED_CONTEXT_FRAMES]
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
            TTSTextFrame(text="Hello "),
            TTSTextFrame(text="Pipecat."),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
            TTSTextFrame(text="How are "),
            TTSTextFrame(text="you?"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            BotStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            BotStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
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
            TTSTextFrame(text="Hello "),
            TTSTextFrame(text="Pipecat."),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(AGGREGATION_SLEEP),
            StartInterruptionFrame(),
            TTSTextFrame(text="How are "),
            TTSTextFrame(text="you?"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            BotStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
            StartInterruptionFrame,
            BotStoppedSpeakingFrame,
            *self.EXPECTED_CONTEXT_FRAMES,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.check_message_multi_content(context, 0, 0, "Hello Pipecat.")
        self.check_message_multi_content(context, 0, 1, "How are you?")


#
# LLMUserContextAggregator, LLMAssistantContextAggregator
#


class TestLLMUserContextAggregator(BaseTestUserContextAggregator, unittest.IsolatedAsyncioTestCase):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = LLMUserContextAggregator


class TestLLMAssistantContextAggregator(
    BaseTestAssistantContextAggreagator, unittest.IsolatedAsyncioTestCase
):
    CONTEXT_CLASS = OpenAILLMContext
    AGGREGATOR_CLASS = LLMAssistantContextAggregator


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
