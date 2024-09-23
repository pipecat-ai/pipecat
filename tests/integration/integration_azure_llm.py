import unittest

import asyncio
import os
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.azure import AzureLLMService

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
)

if __name__ == "__main__":

    @unittest.skip("Skip azure integration test")
    async def test_chat():
        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )
        context = OpenAILLMContext()
        message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Please tell the world hello.", name="system", role="system"
        )
        context.add_message(message)
        frame = OpenAILLMContextFrame(context)
        async for s in llm.process_frame(frame):
            print(s)

    asyncio.run(test_chat())
