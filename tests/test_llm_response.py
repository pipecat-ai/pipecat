#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotInterruptionFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
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
from pipecat.tests.utils import SleepFrame, run_test

AGGREGATION_TIMEOUT = 0.1
AGGREGATION_SLEEP = 0.15
BOT_INTERRUPTION_TIMEOUT = 0.2
BOT_INTERRUPTION_SLEEP = 0.25


class TestLLMUserContextAggreagator(unittest.IsolatedAsyncioTestCase):
    async def test_se(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context)
        frames_to_send = [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_ste(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello!"

    async def test_site(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context)
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
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat!"

    async def test_st1iest2e(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat! ", user_id="cat", timestamp=""),
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
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat! How are you?"

    async def test_siet(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "How are you?"

    async def test_sieit(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "How are you?"

    async def test_set(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "How are you?"

    async def test_seit(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "How are you?"

    async def test_st1et2(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
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
            OpenAILLMContextFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat!"
        assert received_down[-1].context.messages[1]["content"] == "How are you?"

    async def test_set1t2(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat! ", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat! How are you?"

    async def test_siet1it2(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame(text="Hello ", user_id="cat", timestamp=""),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Hello Pipecat! ", user_id="cat", timestamp=""),
            InterimTranscriptionFrame(text="How ", user_id="cat", timestamp=""),
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat! How are you?"

    async def test_t(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [OpenAILLMContextFrame]
        expected_up_frames = [BotInterruptionFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello!"

    async def test_it(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(context, aggregation_timeout=AGGREGATION_TIMEOUT)
        frames_to_send = [
            InterimTranscriptionFrame(text="Hello ", user_id="cat", timestamp=""),
            TranscriptionFrame(text="Hello Pipecat!", user_id="cat", timestamp=""),
            SleepFrame(sleep=AGGREGATION_SLEEP),
        ]
        expected_down_frames = [OpenAILLMContextFrame]
        expected_up_frames = [BotInterruptionFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat!"

    async def test_sie_delay_it(self):
        context = OpenAILLMContext()
        aggregator = LLMUserContextAggregator(
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
            OpenAILLMContextFrame,
        ]
        expected_up_frames = [BotInterruptionFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "How are you?"


class TestLLMAssistantContextAggreagator(unittest.IsolatedAsyncioTestCase):
    async def test_empty(self):
        context = OpenAILLMContext()
        aggregator = LLMAssistantContextAggregator(context)
        frames_to_send = [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]
        expected_down_frames = []
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_single(self):
        context = OpenAILLMContext()
        aggregator = LLMAssistantContextAggregator(context)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [OpenAILLMContextFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat!"

    async def test_multiple(self):
        context = OpenAILLMContext()
        aggregator = LLMAssistantContextAggregator(context, expect_stripped_words=False)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello "),
            TextFrame(text="Pipecat. "),
            TextFrame(text="How are "),
            TextFrame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [OpenAILLMContextFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat. How are you?"

    async def test_multiple_stripped(self):
        context = OpenAILLMContext()
        aggregator = LLMAssistantContextAggregator(context)
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame(text="Hello"),
            TextFrame(text="Pipecat."),
            TextFrame(text="How are"),
            TextFrame(text="you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [OpenAILLMContextFrame]
        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].context.messages[0]["content"] == "Hello Pipecat. How are you?"
