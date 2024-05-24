#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import asyncio
import time
import base64

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    VisionImageRawFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMResponseStartFrame,
    LLMResponseEndFrame,
    LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

from loguru import logger

try:
    from anthropic import AsyncAnthropic
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`. Also, set `ANTHROPIC_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class AnthropicLLMService(LLMService):
    """This class implements inference with Anthropic's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Anthropic Python SDK. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "claude-3-opus-20240229",
            max_tokens: int = 1024):
        super().__init__()
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def _get_messages_from_openai_context(
            self, context: OpenAILLMContext):
        openai_messages = context.get_messages()
        anthropic_messages = []

        for message in openai_messages:
            role = message["role"]
            text = message["content"]
            if role == "system":
                role = "user"
            if message.get("mime_type") == "image/jpeg":
                # vision frame
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                anthropic_messages.append({
                    "role": role,
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": message.get("mime_type"),
                            "data": encoded_image,
                        }
                    }, {
                        "type": "text",
                        "text": text
                    }]
                })
            else:
                # text frame
                anthropic_messages.append({"role": role, "content": content})

        return anthropic_messages

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            start_time = time.time()
            response = await self._client.messages.create(
                messages=messages,
                model=self._model,
                max_tokens=self._max_tokens,
                stream=True)
            logger.debug(f"Anthropic LLM TTFB: {time.time() - start_time}")
            async for event in response:
                # logger.debug(f"Anthropic LLM event: {event}")
                if (event.type == "content_block_delta"):
                    await self.push_frame(LLMResponseStartFrame())
                    await self.push_frame(TextFrame(event.delta.text))
                    await self.push_frame(LLMResponseEndFrame())

        except Exception as e:
            logger.error(f"Exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def x_process_frame(self, frame: Frame, direction: FrameDirection):
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
