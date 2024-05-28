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
    LLMFunctionCallFrame,
    LLMFunctionStartFrame,
    TextFrame
)
from pipecat.utils.test_frame_processor import TestFrameProcessor
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from pipecat.services.openai import OpenAILLMService


if __name__ == "__main__":
    async def test_functions(handle=False):
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

        async def get_weather_from_api(llm, args):
            return json.dumps({"conditions": "nice", "temperature": "75"})

        api_key = os.getenv("OPENAI_API_KEY")

        llm = OpenAILLMService(
            api_key=api_key or "",
            model="gpt-4-1106-preview",
        )

        if (handle):
            llm.register_function("get_current_weather", get_weather_from_api)
            t = TestFrameProcessor([
                LLMFullResponseStartFrame,
                [LLMResponseStartFrame, TextFrame, LLMResponseEndFrame],
                LLMFullResponseEndFrame
            ])
        else:
            t = TestFrameProcessor([
                LLMFullResponseStartFrame,
                LLMFunctionStartFrame,
                LLMFunctionCallFrame,
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
        await test_functions(True)
        await test_functions(False)
        await test_chat()

    asyncio.run(run_tests())
