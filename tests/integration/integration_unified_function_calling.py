import os
from typing import List

import pytest
from dotenv import load_dotenv

from pipecat.adapters.function_schema import FunctionSchema
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame, OpenAILLMService
from pipecat.utils.test_frame_processor import TestFrameProcessor

load_dotenv(override=True)


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


def standard_tools() -> List[FunctionSchema]:
    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location"],
    )
    return [weather_function]


@pytest.mark.asyncio
async def test_unified_function_calling_openai():
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    llm.register_function(None, fetch_weather_from_api)
    t = TestFrameProcessor([LLMFullResponseStartFrame, TextFrame, LLMFullResponseEndFrame])
    llm.link(t)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who can report the weather in any location in the universe. Respond concisely. Your response will be turned into speech so use only simple words and punctuation.",
        },
        {"role": "user", "content": " Start the conversation by introducing yourself."},
    ]
    context = OpenAILLMContext(messages, standard_tools())
    # This is done by default inside the create_context_aggregator
    context.set_llm_adapter(llm.get_llm_adapter())

    frame = OpenAILLMContextFrame(context)

    # This will fail if an exception is raised
    await llm.process_frame(frame, FrameDirection.DOWNSTREAM)
