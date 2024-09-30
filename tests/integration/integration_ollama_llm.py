import unittest

import asyncio
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
)
from pipecat.services.ollama import OLLamaLLMService

if __name__ == "__main__":

    @unittest.skip("Skip azure integration test")
    async def test_chat():
        llm = OLLamaLLMService()
        context = OpenAILLMContext()
        message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Please tell the world hello.", name="system", role="system"
        )
        context.add_message(message)
        frame = OpenAILLMContextFrame(context)
        async for s in llm.process_frame(frame):
            print(s)

    asyncio.run(test_chat())
