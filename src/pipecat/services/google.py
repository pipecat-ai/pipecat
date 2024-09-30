#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
from dataclasses import dataclass
from typing import List

from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMModelUpdateFrame,
    StartInterruptionFrame,
    TextFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class GoogleContextAggregatorPair:
    _user: "GoogleUserContextAggregator"
    _assistant: "GoogleAssistantContextAggregator"

    def user(self) -> "GoogleUserContextAggregator":
        return self._user

    def assistant(self) -> "GoogleAssistantContextAggregator":
        return self._assistant


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        gai.configure(api_key=api_key)
        self._create_client(model)

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> GoogleContextAggregatorPair:
        user = GoogleUserContextAggregator(context)
        assistant = GoogleAssistantContextAggregator(user)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)

    def _create_client(self, model: str):
        self.set_model_name(model)
        self._client = gai.GenerativeModel(model)

    def _get_messages_from_openai_context(self, context: OpenAILLMContext) -> List[glm.Content]:
        openai_messages = context.get_messages()
        google_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            if isinstance(content, list):
                parts = []
                for item in content:
                    if item["type"] == "text":
                        parts.append(glm.Part(text=item["text"]))
                    elif item["type"] == "image_url":
                        image_data = item["image_url"]["url"].split(",")[1]
                        parts.append(
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type="image/jpeg", data=base64.b64decode(image_data)
                                )
                            )
                        )
            else:
                parts = [glm.Part(text=content)]

            google_messages.append(glm.Content(role=role, parts=parts))

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()

            response = self._client.generate_content(messages, stream=True)

            await self.stop_ttfb_metrics()

            async for chunk in self._async_generator_wrapper(response):
                try:
                    text = chunk.text
                    await self.push_frame(TextFrame(text))
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}."
                        )
                    else:
                        logger.exception(f"{self} error: {e}")

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = GoogleLLMContext.from_openai_context(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = GoogleLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = GoogleLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self._create_client(frame.model)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class GoogleLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self._user_image_request_context = {}

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        return cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )

    @classmethod
    def from_messages(cls, messages: List[dict]) -> "GoogleLLMContext":
        return cls(messages=messages)

    @classmethod
    def from_image_frame(cls, frame: VisionImageRawFrame) -> "GoogleLLMContext":
        context = cls()
        context.add_image_frame_message(
            format=frame.format, size=frame.size, image=frame.image, text=frame.text
        )
        return context

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
        if text:
            content.append({"type": "text", "text": text})
        self.add_message({"role": "user", "content": content})


class GoogleUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext | GoogleLLMContext):
        super().__init__(context=context)

        if isinstance(context, OpenAILLMContext):
            self._context = GoogleLLMContext.from_openai_context(context)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, UserImageRequestFrame):
                if frame.context:
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}"
                        )
                        del self._context._user_image_request_context[frame.user_id]
                else:
                    if frame.user_id in self._context._user_image_request_context:
                        del self._context._user_image_request_context[frame.user_id]
            elif isinstance(frame, UserImageRawFrame):
                text = self._context._user_image_request_context.get(frame.user_id) or ""
                if text:
                    del self._context._user_image_request_context[frame.user_id]

                # Handle the case where frame.format might be None
                image_format = frame.format or "JPEG"  # Default to JPEG if format is None

                self._context.add_image_frame_message(
                    format=image_format, size=frame.size, image=frame.image, text=text
                )
                await self.push_context_frame()
        except Exception as e:
            logger.error(f"Error processing frame: {e}")


class GoogleAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: GoogleUserContextAggregator):
        super().__init__(context=user_context_aggregator._context)
        self._user_context_aggregator = user_context_aggregator
        self._function_call_in_progress = None
        self._function_call_result = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            self._function_call_in_progress = None
            self._function_call_result = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = frame
        elif isinstance(frame, FunctionCallResultFrame):
            if (
                self._function_call_in_progress
                and self._function_call_in_progress.tool_call_id == frame.tool_call_id
            ):
                self._function_call_in_progress = None
                self._function_call_result = frame
                await self._push_aggregation()
            else:
                logger.warning(
                    "FunctionCallResultFrame tool_call_id != InProgressFrame tool_call_id"
                )
                self._function_call_in_progress = None
                self._function_call_result = None

    async def _push_aggregation(self):
        if not (self._aggregation or self._function_call_result):
            return

        run_llm = False

        aggregation = self._aggregation
        self._aggregation = ""

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "content": aggregation,
                            "function_call": {
                                "name": frame.function_name,
                                "arguments": json.dumps(frame.arguments),
                            },
                        }
                    )
                    self._context.add_message(
                        {
                            "role": "function",
                            "content": json.dumps(frame.result),
                            "name": frame.function_name,
                        }
                    )
                    run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
