import os
from unittest.mock import AsyncMock

import pytest
from dotenv import load_dotenv

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame, OpenAILLMService
from pipecat.utils.test_frame_processor import TestFrameProcessor

load_dotenv(override=True)


def standard_tools() -> ToolsSchema:
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
    tools_def = ToolsSchema(standard_tools=[weather_function])
    return tools_def


async def _test_llm_function_calling(llm: LLMService):
    # Create an AsyncMock for the function
    mock_fetch_weather = AsyncMock()

    llm.register_function(None, mock_fetch_weather)
    t = TestFrameProcessor([LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame])
    llm.link(t)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who can report the weather in any location in the universe. Respond concisely. Your response will be turned into speech so use only simple words and punctuation.",
        },
        {"role": "user", "content": " How is the weather today in San Francisco, California?"},
    ]
    context = OpenAILLMContext(messages, standard_tools())
    # This is done by default inside the create_context_aggregator
    context.set_llm_adapter(llm.get_llm_adapter())

    frame = OpenAILLMContextFrame(context)

    # This will fail if an exception is raised
    await llm.process_frame(frame, FrameDirection.DOWNSTREAM)

    # Assert that the mock function was called
    mock_fetch_weather.assert_called_once()


@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY is not set")
@pytest.mark.asyncio
async def test_unified_function_calling_openai():
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    # This will fail if an exception is raised
    await _test_llm_function_calling(llm)


@pytest.mark.skipif(os.getenv("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY is not set")
@pytest.mark.asyncio
async def test_unified_function_calling_gemini():
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
    # This will fail if an exception is raised
    await _test_llm_function_calling(llm)


@pytest.mark.skipif(os.getenv("ANTHROPIC_API_KEY") is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test_unified_function_calling_anthropic():
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20240620"
    )
    # This will fail if an exception is raised
    await _test_llm_function_calling(llm)
