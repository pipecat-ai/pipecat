import asyncio
import os
from dailyai.pipeline.openai_frames import OpenAILLMContextFrame
from dailyai.services.openai_llm_context import OpenAILLMContext

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from dailyai.services.openai_api_llm_service import BaseOpenAILLMService

if __name__ == "__main__":
    async def test_functions():
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

        api_key = os.getenv("OPENAI_API_KEY")

        llm = BaseOpenAILLMService(
            api_key=api_key or "",
            model="gpt-4-1106-preview",
        )
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
        async for s in llm.process_frame(frame):
            print(s)

    async def test_chat():
        api_key = os.getenv("OPENAI_API_KEY")

        llm = BaseOpenAILLMService(
            api_key=api_key or "",
            model="gpt-4-1106-preview",
        )
        context = OpenAILLMContext()
        message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Please tell the world hello.", name="system", role="system")
        context.add_message(message)
        frame = OpenAILLMContextFrame(context)
        async for s in llm.process_frame(frame):
            print(s)

    async def run_tests():
        await test_functions()
        await test_chat()

    asyncio.run(run_tests())
