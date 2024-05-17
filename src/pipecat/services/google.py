import google.generativeai as gai
import google.ai.generativelanguage as glm
import os
import asyncio

from typing import List

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    VisionImageRawFrame,
    LLMMessagesFrame,
    LLMResponseStartFrame,
    LLMResponseEndFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

from loguru import logger


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(self, model="gemini-1.5-flash-latest", api_key=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        gai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self.create_client()

    def create_client(self):
        self._client = gai.GenerativeModel(self.model)

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
                        data=message["data"]
                    )))
            google_messages.append({"role": role, "parts": parts})

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        try:
            messages = self._get_messages_from_openai_context(context)

            await self.push_frame(LLMResponseStartFrame())
            response = self._client.generate_content(messages, stream=True)

            async for chunk in self._async_generator_wrapper(response):
                logger.debug(f"Pushing inference text: {chunk.text}")
                await self.push_frame(TextFrame(chunk.text))

            await self.push_frame(LLMResponseEndFrame())
        except Exception as e:
            logger.error(f"Exception: {e}")

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
