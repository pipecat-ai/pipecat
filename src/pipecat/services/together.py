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
    UserImageRequestFrame,
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
    from together import AsyncTogether
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Together.ai, you need to `pip install pipecat-ai[together]`. Also, set `TOGETHER_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


@dataclass
class TogetherContextAggregatorPair:
    _user: 'TogetherUserContextAggregator'
    _assistant: 'TogetherAssistantContextAggregator'

    def user(self) -> 'TogetherUserContextAggregator':
        return self._user

    def assistant(self) -> 'TogetherAssistantContextAggregator':
        return self._assistant


class TogetherLLMService(LLMService):
    """This class implements inference with Together's Llama 3.1 models
    """

    def __init__(
            self,
            *,
            api_key: str,
            model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            max_tokens: int = 4096,
            **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncTogether(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> TogetherContextAggregatorPair:
        user = TogetherUserContextAggregator(context)
        assistant = TogetherAssistantContextAggregator(user)
        return TogetherContextAggregatorPair(
            _user=user,
            _assistant=assistant
        )

    async def _process_context(self, context: OpenAILLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

            await self.start_ttfb_metrics()

            stream = await self._client.chat.completions.create(
                messages=context.messages,
                model=self._model,
                max_tokens=self._max_tokens,
                stream=True,
            )

            got_first_chunk = False

            # Function calling. We should be able to prompt Llama 3.1 to always return either plain
            # text or a function call. However, occasionally we see a function call after plain text.
            # Try to account for that.
            most_recent_chunk_was_function_call_start_char = False  # function call start char is '<'
            accumulating_function_call = False
            function_call_accumulator = ""

            async for chunk in stream:
                # logger.debug(f"Together LLM event: {chunk}")
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
                        text = chunk.choices[0].delta.content
                        if most_recent_chunk_was_function_call_start_char:
                            most_recent_chunk_was_function_call_start_char = False
                            if text == "function":
                                accumulating_function_call = True
                                function_call_accumulator = "<function"
                            else:
                                await self.push_frame("<" + TextFrame(chunk.choices[0].delta.content))
                        elif text == '<':
                            most_recent_chunk_was_function_call_start_char = True
                        else:
                            await self.push_frame(TextFrame(chunk.choices[0].delta.content))

                if chunk.choices[0].finish_reason:
                    if accumulating_function_call:
                        await self._extract_function_call(context, function_call_accumulator)
                    elif most_recent_chunk_was_function_call_start_char:
                        await self.push_frame(TextFrame("<"))

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
            context = TogetherLLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _extract_function_call(self, context, function_call_accumulator):
        # logger.debug(f"Extracting function call: {function_call_accumulator}")
        context.add_message({"role": "assistant", "content": function_call_accumulator})

        # Function format regex. Llama 3.1 sometimes adds an extra " or space just before the
        # </function> tag. This regexp just ignores the extra characters if they are there. (That's
        # the [\s"]? part of the regex.) Occasionally the </function> close tag is also missing.
        function_regex = r'<function=(\w+)>(.*?)<\/function>|<function=(\w+)>(.*)'
        match = re.search(function_regex, function_call_accumulator)
        if match:
            function_name = ""
            args_string = ""
            if match.group(1):  # Case with closing tag
                function_name = match.group(1)
                args_string = match.group(2)
            else:  # Case without closing tag
                function_name = match.group(3)
                args_string = match.group(4)

            try:
                args_string = re.sub(r'[\s"]+$', '', args_string)
                arguments = json.loads(args_string) if args_string else ""
                await self.call_function(context=context,
                                         tool_call_id=str(uuid.uuid4()),
                                         function_name=function_name,
                                         arguments=arguments)
                return
            except json.JSONDecodeError as error:
                # We get here if the LLM returns a function call with invalid JSON arguments. This could happen
                # because of LLM non-determinism, or maybe more often because of user error in the prompt.
                # Should we do anything more than log a warning?
                logger.debug(
                    f"Error parsing function arguments: {error} - {function_call_accumulator}")


class TogetherLLMContext(OpenAILLMContext):
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
    def from_messages(cls, messages: List[dict]) -> "TogetherLLMContext":
        return cls(messages=messages)

    def add_message(self, message):
        try:
            self.messages.append(message)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def get_messages_for_logging(self) -> str:
        return json.dumps(self.messages)


class TogetherUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext | TogetherLLMContext):
        super().__init__(context=context)

        if isinstance(context, OpenAILLMContext):
            self._context = TogetherLLMContext.from_openai_context(context)

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


class TogetherAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: TogetherUserContextAggregator):
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
                    # Together expects the content here to be a string, so stringify it
                    "content": str(frame.result)
                })
                run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_messages_frame()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
