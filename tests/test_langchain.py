import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import FakeStreamingListLLM

from pipecat.frames.frames import (StopTaskFrame, TranscriptionFrame,
                                   UserStartedSpeakingFrame,
                                   UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.logger import FrameLogger
from pipecat.services.langchain import LangchainProcessor


@pytest.fixture
def fake_llm():
    responses = ["Hello dear human"]
    return FakeStreamingListLLM(responses=responses)


@pytest.mark.asyncio
async def test_langchain(fake_llm: FakeStreamingListLLM):
    fl_in = FrameLogger("Inner")
    fl_out = FrameLogger("Outer")

    messages = [("system", "Say hello to {name}"), ("human", "{input}")]
    prompt = ChatPromptTemplate.from_messages(messages).partial(name="Thomas")
    chain = prompt | fake_llm
    proc = LangchainProcessor(chain=chain)

    tma_in = LLMUserResponseAggregator(messages)
    tma_out = LLMAssistantResponseAggregator(messages)

    pipeline = Pipeline(
        [
            fl_in,
            tma_in,
            proc,
            tma_out,
            fl_out,
        ]
    )

    task = PipelineTask(pipeline)
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
