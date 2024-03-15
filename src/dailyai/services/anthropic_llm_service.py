import asyncio
import os
from typing import AsyncGenerator
from anthropic import AsyncAnthropic
from dailyai.pipeline.frames import Frame, LLMMessagesQueueFrame, TextFrame

from dailyai.services.ai_services import LLMService


class AnthropicLLMService(LLMService):

    def __init__(self, api_key, model="claude-3-opus-20240229", max_tokens=1024):
        super().__init__()
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if not isinstance(frame, LLMMessagesQueueFrame):
            yield frame

        stream = await self.client.messages.create(
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, Claude",
                }
            ],
            model=self.model,
            stream=True,
        )
        async for event in stream:
            if event.type == "content_block_delta":
                yield TextFrame(event.delta.text)


async def test():
    service = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ]
    async for frame in service.process_frame(LLMMessagesQueueFrame(messages)):
        print(frame)


if __name__=="__main__":
    asyncio.run(test())
