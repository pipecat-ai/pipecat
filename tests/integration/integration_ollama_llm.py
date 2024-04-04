import asyncio
from dailyai.pipeline.openai_frames import OpenAILLMContextFrame
from dailyai.services.openai_llm_context import OpenAILLMContext

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
)
from dailyai.services.ollama_ai_services import OLLamaLLMService

if __name__ == "__main__":
    async def test_chat():
        llm = OLLamaLLMService()
        context = OpenAILLMContext()
        message: ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam(
            content="Please tell the world hello.", name="system", role="system")
        context.add_message(message)
        frame = OpenAILLMContextFrame(context)
        async for s in llm.process_frame(frame):
            print(s)

    asyncio.run(test_chat())
