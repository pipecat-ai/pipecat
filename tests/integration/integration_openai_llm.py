import asyncio
import json
import os
from typing import List

from pipecat.services.openai import OpenAILLMContextFrame, OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame
)
from pipecat.utils.test_frame_processor import TestFrameProcessor
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from pipecat.services.openai import OpenAILLMService

tools = [
    ChatCompletionToolParam(
        type="function",
        function={
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": [
                            "celsius",
                            "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": [
                    "location",
                    "format"],
            },
        })]

if __name__ == "__main__":
    async def test_simple_functions():

        async def get_weather_from_api(llm, args):
            return json.dumps({"conditions": "nice", "temperature": "75"})

        api_key = os.getenv("OPENAI_API_KEY")

        llm = OpenAILLMService(
            api_key=api_key or "",
            model="gpt-4-1106-preview",
        )

        llm.register_function("get_current_weather", get_weather_from_api)
        t = TestFrameProcessor([
            LLMFullResponseStartFrame,
            [LLMResponseStartFrame, TextFrame, LLMResponseEndFrame],
            LLMFullResponseEndFrame
        ])
        llm.link(t)

        context = OpenAILLMContext(tools=tools)
        system_message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Ask the user to ask for a weather report", name="system", role="system"
        )
        user_message: ChatCompletionUserMessageParam = ChatCompletionUserMessageParam(
            content="Could you tell me the weather for Boulder, Colorado",
            name="user",
            role="user",
        )
        context.add_message(system_message)
        context.add_message(user_message)
        frame = OpenAILLMContextFrame(context)
        await llm.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_advanced_functions():

        async def get_weather_from_api(llm, args):
            return [{"role": "system", "content": "The user has asked for live weather. Respond by telling them we don't currently support live weather for that area, but it's coming soon."}]

        api_key = os.getenv("OPENAI_API_KEY")

        llm = OpenAILLMService(
            api_key=api_key or "",
            model="gpt-4-1106-preview",
        )

        llm.register_function("get_current_weather", get_weather_from_api)
        t = TestFrameProcessor([
            LLMFullResponseStartFrame,
            [LLMResponseStartFrame, TextFrame, LLMResponseEndFrame],
            LLMFullResponseEndFrame
        ])
        llm.link(t)

        context = OpenAILLMContext(tools=tools)
        system_message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Ask the user to ask for a weather report", name="system", role="system"
        )
        user_message: ChatCompletionUserMessageParam = ChatCompletionUserMessageParam(
            content="Could you tell me the weather for Boulder, Colorado",
            name="user",
            role="user",
        )
        context.add_message(system_message)
        context.add_message(user_message)
        frame = OpenAILLMContextFrame(context)
        await llm.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_chat():
        api_key = os.getenv("OPENAI_API_KEY")
        t = TestFrameProcessor([
            LLMFullResponseStartFrame,
            [LLMResponseStartFrame, TextFrame, LLMResponseEndFrame],
            LLMFullResponseEndFrame
        ])
        llm = OpenAILLMService(
            api_key=api_key or "",
            model="gpt-4o",
        )
        llm.link(t)
        context = OpenAILLMContext()
        message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Please tell the world hello.", name="system", role="system")
        context.add_message(message)
        frame = OpenAILLMContextFrame(context)
        await llm.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def run_tests():
        await test_simple_functions()
        await test_advanced_functions()
        await test_chat()

    asyncio.run(run_tests())
