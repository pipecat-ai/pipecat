from typing import AsyncGenerator
from dailyai.pipeline.frames import Frame, LLMMessagesFrame, TextFrame

from dailyai.services.ai_services import LLMService

try:
    from anthropic import AsyncAnthropic
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use Anthropic, you need to `pip install dailyai[anthropic]`. Also, set `ANTHROPIC_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class AnthropicLLMService(LLMService):

    def __init__(
            self,
            api_key,
            model="claude-3-opus-20240229",
            max_tokens=1024):
        super().__init__()
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if not isinstance(frame, LLMMessagesFrame):
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
