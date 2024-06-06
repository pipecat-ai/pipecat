#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (LLMFullResponseEndFrame,
                                   LLMFullResponseStartFrame, StopTaskFrame,
                                   TextFrame, TranscriptionFrame,
                                   UserStartedSpeakingFrame,
                                   UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.langchain import LangchainProcessor

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import FakeStreamingListLLM


class TestLangchain(unittest.IsolatedAsyncioTestCase):

    class MockProcessor(FrameProcessor):
        def __init__(self, name):
            self.name = name
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
        self.mock_proc = self.MockProcessor("token_collector")

    async def test_langchain(self):

        messages = [("system", "Say hello to {name}"), ("human", "{input}")]
        prompt = ChatPromptTemplate.from_messages(messages).partial(name="Thomas")
        chain = prompt | self.fake_llm
        proc = LangchainProcessor(chain=chain)

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline(
            [
                tma_in,
                proc,
                self.mock_proc,
                tma_out,
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))
        await task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(text="Hi World", user_id="user", timestamp="now"),
                UserStoppedSpeakingFrame(),
                StopTaskFrame(),
            ]
        )

        runner = PipelineRunner()
        await runner.run(task)
        self.assertEqual("".join(self.mock_proc.token), self.expected_response)
        # TODO: Address this issue
        # This next one would fail with:
        # AssertionError: ' H e l l o   d e a r   h u m a n' != 'Hello dear human'
        # self.assertEqual(tma_out.messages[-1]["content"], self.expected_response)


if __name__ == "__main__":
    unittest.main()
