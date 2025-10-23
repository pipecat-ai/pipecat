#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import FakeStreamingListLLM

from pipecat.frames.frames import (
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestLangchain(unittest.IsolatedAsyncioTestCase):
    class MockProcessor(FrameProcessor):
        def __init__(self, name):
            super().__init__(name=name)
            self.token: list[str] = []
            # Start collecting tokens when we see the start frame
            self.start_collecting = False

        def __str__(self):
            return self.name

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self.start_collecting = True
            elif isinstance(frame, TextFrame) and self.start_collecting:
                self.token.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self.start_collecting = False

            await self.push_frame(frame, direction)

    def setUp(self):
        self.expected_response = "Hello dear human"
        self.fake_llm = FakeStreamingListLLM(responses=[self.expected_response])

    async def test_langchain(self):
        messages = [("system", "Say hello to {name}"), ("human", "{input}")]
        prompt = ChatPromptTemplate.from_messages(messages).partial(name="Thomas")
        chain = prompt | self.fake_llm
        proc = LangchainProcessor(chain=chain)
        self.mock_proc = self.MockProcessor("token_collector")

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context, assistant_params=LLMAssistantAggregatorParams(expect_stripped_words=False)
        )

        pipeline = Pipeline(
            [context_aggregator.user(), proc, self.mock_proc, context_aggregator.assistant()]
        )

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hi World", user_id="user", timestamp="now"),
            SleepFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        self.assertEqual("".join(self.mock_proc.token), self.expected_response)
        self.assertEqual(
            context_aggregator.assistant().messages[-1]["content"], self.expected_response
        )
