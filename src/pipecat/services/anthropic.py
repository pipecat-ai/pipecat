#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService

from loguru import logger

try:
    from anthropic import AsyncAnthropic
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`. Also, set `ANTHROPIC_API_KEY` environment variable.")
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMMessagesFrame):
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
                    await self.push_frame(TextFrame(event.delta.text))
        else:
            await self.push_frame(frame, direction)
