#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class OpenAIUnhandledFunctionException(Exception):
    pass


class BaseOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    class InputParams(BaseModel):
        frequency_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        presence_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        seed: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignored right now.
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        max_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        max_completion_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Mapping[str, str] | None = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self.set_model_name(model)
        self._client = self.create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
            default_headers=default_headers,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"{self}: Generating chat [{context.get_messages_for_logging()}]")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)

        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.tool_calls:
                # We're streaming the LLM response to enable the fastest response times.
                # For text, we just yield each chunk as we receive it and count on consumers
                # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                #
                # If the LLM is a function call, we'll do some coalescing here.
                # If the response contains a function name, we'll yield a frame to tell consumers
                # that they can start preparing to call the function with that name.
                # We accumulate all the arguments for the rest of the streamed response, then when
                # the response is done, we package up all the arguments and the function name and
                # yield a frame containing the function name and the arguments.

                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
                    tool_id_list.append(tool_call_id)
                    function_name = ""
                    arguments = ""
                    tool_call_id = ""
                    func_idx += 1
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            for index, (function_name, arguments, tool_id) in enumerate(
                zip(functions_list, arguments_list, tool_id_list), start=1
            ):
                if self.has_function(function_name):
                    run_llm = False
                    arguments = json.loads(arguments)
                    await self.call_function(
                        context=context,
                        function_name=function_name,
                        arguments=arguments,
                        tool_call_id=tool_id,
                        run_llm=run_llm,
                    )
                else:
                    raise OpenAIUnhandledFunctionException(
                        f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                    )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext()
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except httpx.TimeoutException:
                await self._call_event_handler("on_completion_timeout")
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
