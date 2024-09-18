#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import io
import copy
from typing import List, Optional
from dataclasses import dataclass
from asyncio import CancelledError
import re
import uuid


from pipecat.frames.frames import (
    Frame,
    LLMModelUpdateFrame,
    TextFrame,
    VisionImageRawFrame,
    UserImageRequestFrame,
    UserImageRawFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    FunctionCallResultFrame,
    FunctionCallInProgressFrame,
    StartInterruptionFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator

from loguru import logger


try:
    from cohere import AsyncClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cohere, you need to `pip install pipecat-ai[cohere]`. Also, set `COHERE_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


@dataclass
class CohereContextAggregatorPair:
    _user: 'CohereUserContextAggregator'
    _assistant: 'CohereAssistantContextAggregator'

    def user(self) -> 'CohereUserContextAggregator':
        return self._user

    def assistant(self) -> 'CohereAssistantContextAggregator':
        return self._assistant


class CohereLLMService(LLMService):
    """This class implements inference with Cohere's models
    """

    def __init__(
            self,
            *,
            api_key: str,
            model: str = "command-r-plus",
            max_tokens: int = 4096,
            **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncClient(
            api_key,
            model,
        )
        self._model = model
        self._max_tokens = max_tokens

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> CohereContextAggregatorPair:
        user = CohereUserContextAggregator(context)
        assistant = CohereAssistantContextAggregator(user)
        return CohereContextAggregatorPair(
            _user=user,
            _assistant=assistant
        )

    async def _process_context(self, context: OpenAILLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

            await self.start_ttfb_metrics()

            stream = await self._client.chat_stream(
                chat_history=context.messages,
                model=self._model,
                max_tokens=self._max_tokens,
                stream=True,
            )

            # Function calling
            got_first_chunk = False
            accumulating_function_call = False
            function_call_accumulator = ""

            async for chunk in stream:
                logger.debug(f"Cohere LLM event: {chunk}")
                if chunk.usage:
                    tokens = {
                        "processor": self.name,
                        "model": self._model,
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
                    await self.start_llm_usage_metrics(tokens)

                if len(chunk.choices) == 0:
                    continue

                if not got_first_chunk:
                    await self.stop_ttfb_metrics()
                    if chunk.choices[0].delta.content:
                        got_first_chunk = True
                        if chunk.choices[0].delta.content[0] == "<":
                            accumulating_function_call = True

                if chunk.choices[0].delta.content:
                    if accumulating_function_call:
                        function_call_accumulator += chunk.choices[0].delta.content
                    else:
                        await self.push_frame(TextFrame(chunk.choices[0].delta.content))



        except CancelledError as e:
            # todo: implement token counting estimates for use when the user interrupts a long generation
            # we do this in the anthropic.py service
            raise
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = CohereLLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

class CohereLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: list[dict] | None = None,
    ):
        super().__init__(messages=messages)

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        self = cls(
            messages=openai_context.messages,
        )
        return self

    @classmethod
    def from_messages(cls, messages: List[dict]) -> "CohereLLMContext":
        return cls(messages=messages)

    def add_message(self, message):
        try:
            self.messages.append(message)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def get_messages_for_logging(self) -> str:
        return json.dumps(self.messages)


class CohereUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext | CohereLLMContext):
        super().__init__(context=context)

        if isinstance(context, OpenAILLMContext):
            self._context = CohereLLMContext.from_openai_context(context)

    async def push_messages_frame(self):
        frame = OpenAILLMContextFrame(self._context)
        await self.push_frame(frame)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # Our parent method has already called push_frame(). So we can't interrupt the
        # flow here and we don't need to call push_frame() ourselves. Possibly something
        # to talk through (tagging @aleix). At some point we might need to refactor these
        # context aggregators.
        try:
            if isinstance(frame, UserImageRequestFrame):
                # The LLM sends a UserImageRequestFrame upstream. Cache any context provided with
                # that frame so we can use it when we assemble the image message in the assistant
                # context aggregator.
                if (frame.context):
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}")
                        del self._context._user_image_request_context[frame.user_id]
                else:
                    if frame.user_id in self._context._user_image_request_context:
                        del self._context._user_image_request_context[frame.user_id]
        except Exception as e:
            logger.error(f"Error processing frame: {e}")



class CohereAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: CohereUserContextAggregator):
        super().__init__(context=user_context_aggregator._context)
        self._user_context_aggregator = user_context_aggregator
        self._function_call_in_progress = None
        self._function_call_result = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_call_in_progress = None
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = frame
        elif isinstance(frame, FunctionCallResultFrame):
            if self._function_call_in_progress and self._function_call_in_progress.tool_call_id == frame.tool_call_id:
                self._function_call_in_progress = None
                self._function_call_result = frame
                await self._push_aggregation()
            else:
                logger.warning(
                    f"FunctionCallResultFrame tool_call_id does not match FunctionCallInProgressFrame tool_call_id")
                self._function_call_in_progress = None
                self._function_call_result = None

    def add_message(self, message):
        self._user_context_aggregator.add_message(message)

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
                self._context.add_message({
                    "role": "tool",
                    # Cohere expects the content here to be a string, so stringify it
                    "content": str(frame.result)
                })
                run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_messages_frame()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
