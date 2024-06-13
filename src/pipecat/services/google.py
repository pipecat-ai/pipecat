#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import List

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
    import google.generativeai as gai
    import google.ai.generativelanguage as glm
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        gai.configure(api_key=api_key)
        self._client = gai.GenerativeModel(model)

    def can_generate_metrics(self) -> bool:
        return True

    def _get_messages_from_openai_context(
            self, context: OpenAILLMContext) -> List[glm.Content]:
        openai_messages = context.get_messages()
        google_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            parts = [glm.Part(text=content)]
            if "mime_type" in message:
                parts.append(
                    glm.Part(inline_data=glm.Blob(
                        mime_type=message["mime_type"],
                        data=message["data"].getvalue()
                    )))
            google_messages.append({"role": role, "parts": parts})

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()

            response = self._client.generate_content(messages, stream=True)

            await self.stop_ttfb_metrics()

            async for chunk in self._async_generator_wrapper(response):
                try:
                    text = chunk.text
                    await self.push_frame(LLMResponseStartFrame())
                    await self.push_frame(TextFrame(text))
                    await self.push_frame(LLMResponseEndFrame())
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}.")
                    else:
                        logger.error(f"{self} error: {e}")

        except Exception as e:
            logger.error(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

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
