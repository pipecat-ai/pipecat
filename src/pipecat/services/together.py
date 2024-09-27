#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import re
import uuid
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    TextFrame,
    UserImageRequestFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
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
    from together import AsyncTogether
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Together.ai, you need to `pip install pipecat-ai[together]`. Also, set `TOGETHER_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class TogetherContextAggregatorPair:
    _user: "TogetherUserContextAggregator"
    _assistant: "TogetherAssistantContextAggregator"

    def user(self) -> "TogetherUserContextAggregator":
        return self._user

    def assistant(self) -> "TogetherAssistantContextAggregator":
        return self._assistant


class TogetherLLMService(LLMService):
    """This class implements inference with Together's Llama 3.1 models"""

    class InputParams(BaseModel):
        frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
        max_tokens: Optional[int] = Field(default=4096, ge=1)
        presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = AsyncTogether(api_key=api_key)
        self.set_model_name(model)
        self._max_tokens = params.max_tokens
        self._frequency_penalty = params.frequency_penalty
        self._presence_penalty = params.presence_penalty
        self._temperature = params.temperature
        self._top_k = params.top_k
        self._top_p = params.top_p
        self._extra = params.extra if isinstance(params.extra, dict) else {}

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> TogetherContextAggregatorPair:
        user = TogetherUserContextAggregator(context)
        assistant = TogetherAssistantContextAggregator(user)
        return TogetherContextAggregatorPair(_user=user, _assistant=assistant)

    async def set_frequency_penalty(self, frequency_penalty: float):
        logger.debug(f"Switching LLM frequency_penalty to: [{frequency_penalty}]")
        self._frequency_penalty = frequency_penalty

    async def set_max_tokens(self, max_tokens: int):
        logger.debug(f"Switching LLM max_tokens to: [{max_tokens}]")
        self._max_tokens = max_tokens

    async def set_presence_penalty(self, presence_penalty: float):
        logger.debug(f"Switching LLM presence_penalty to: [{presence_penalty}]")
        self._presence_penalty = presence_penalty

    async def set_temperature(self, temperature: float):
        logger.debug(f"Switching LLM temperature to: [{temperature}]")
        self._temperature = temperature

    async def set_top_k(self, top_k: float):
        logger.debug(f"Switching LLM top_k to: [{top_k}]")
        self._top_k = top_k

    async def set_top_p(self, top_p: float):
        logger.debug(f"Switching LLM top_p to: [{top_p}]")
        self._top_p = top_p

    async def set_extra(self, extra: Dict[str, Any]):
        logger.debug(f"Switching LLM extra to: [{extra}]")
        self._extra = extra

    async def _process_context(self, context: OpenAILLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

            await self.start_ttfb_metrics()

            params = {
                "messages": context.messages,
                "model": self.model_name,
                "max_tokens": self._max_tokens,
                "stream": True,
                "frequency_penalty": self._frequency_penalty,
                "presence_penalty": self._presence_penalty,
                "temperature": self._temperature,
                "top_k": self._top_k,
                "top_p": self._top_p,
            }

            params.update(self._extra)

            stream = await self._client.chat.completions.create(**params)

            # Function calling
            got_first_chunk = False
            accumulating_function_call = False
            function_call_accumulator = ""

            async for chunk in stream:
                # logger.debug(f"Together LLM event: {chunk}")
                if chunk.usage:
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
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

                if chunk.choices[0].finish_reason == "eos" and accumulating_function_call:
                    await self._extract_function_call(context, function_call_accumulator)

        except CancelledError:
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
        elif isinstance(frame, LLMUpdateSettingsFrame):
            if frame.model is not None:
                logger.debug(f"Switching LLM model to: [{frame.model}]")
                self.set_model_name(frame.model)
            if frame.frequency_penalty is not None:
                await self.set_frequency_penalty(frame.frequency_penalty)
            if frame.max_tokens is not None:
                await self.set_max_tokens(frame.max_tokens)
            if frame.presence_penalty is not None:
                await self.set_presence_penalty(frame.presence_penalty)
            if frame.temperature is not None:
                await self.set_temperature(frame.temperature)
            if frame.top_k is not None:
                await self.set_top_k(frame.top_k)
            if frame.top_p is not None:
                await self.set_top_p(frame.top_p)
            if frame.extra:
                await self.set_extra(frame.extra)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _extract_function_call(self, context, function_call_accumulator):
        context.add_message({"role": "assistant", "content": function_call_accumulator})

        function_regex = r"<function=(\w+)>(.*?)</function>"
        match = re.search(function_regex, function_call_accumulator)
        if match:
            function_name, args_string = match.groups()
            try:
                arguments = json.loads(args_string)
                await self.call_function(
                    context=context,
                    tool_call_id=str(uuid.uuid4()),
                    function_name=function_name,
                    arguments=arguments,
                )
                return
            except json.JSONDecodeError as error:
                # We get here if the LLM returns a function call with invalid JSON arguments. This could happen
                # because of LLM non-determinism, or maybe more often because of user error in the prompt.
                # Should we do anything more than log a warning?
                logger.debug(f"Error parsing function arguments: {error}")


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
        except Exception as e:
            logger.error(f"Error processing frame: {e}")


#
# Claude returns a text content block along with a tool use content block. This works quite nicely
# with streaming. We get the text first, so we can start streaming it right away. Then we get the
# tool_use block. While the text is streaming to TTS and the transport, we can run the tool call.
#
# But Claude is verbose. It would be nice to come up with prompt language that suppresses Claude's
# chattiness about it's tool thinking.
#


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
            if (
                self._function_call_in_progress
                and self._function_call_in_progress.tool_call_id == frame.tool_call_id
            ):
                self._function_call_in_progress = None
                self._function_call_result = frame
                await self._push_aggregation()
            else:
                logger.warning(
                    "FunctionCallResultFrame tool_call_id does not match FunctionCallInProgressFrame tool_call_id"
                )
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
                self._context.add_message(
                    {
                        "role": "tool",
                        # Together expects the content here to be a string, so stringify it
                        "content": str(frame.result),
                    }
                )
                run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_messages_frame()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
