#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base LLM service implementation for services that use the AsyncOpenAI client."""

import asyncio
import base64
import json
from typing import Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import BaseModel, Field

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.tracing.service_decorators import traced_llm


class BaseOpenAILLMService(LLMService):
    """Base class for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame or LLMContextFrame frames,
    which contain a reference to an OpenAILLMContext or LLMContext object. The
    context defines what is sent to the LLM for completion, including user,
    assistant, and system messages, as well as tool choices and function call
    configurations.
    """

    class InputParams(BaseModel):
        """Input parameters for OpenAI model configuration.

        Parameters:
            frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0).
            presence_penalty: Penalty for new tokens (-2.0 to 2.0).
            seed: Random seed for deterministic outputs.
            temperature: Sampling temperature (0.0 to 2.0).
            top_k: Top-k sampling parameter (currently ignored by OpenAI).
            top_p: Top-p (nucleus) sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens in response (deprecated, use max_completion_tokens).
            max_completion_tokens: Maximum completion tokens to generate.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            extra: Additional model-specific parameters.
        """

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
        service_tier: Optional[str] = Field(default_factory=lambda: NOT_GIVEN)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[InputParams] = None,
        retry_timeout_secs: Optional[float] = 5.0,
        retry_on_timeout: Optional[bool] = False,
        **kwargs,
    ):
        """Initialize the BaseOpenAILLMService.

        Args:
            model: The OpenAI model name to use (e.g., "gpt-4.1", "gpt-4o").
            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            params: Input parameters for model configuration and behavior.
            retry_timeout_secs: Request timeout in seconds. Defaults to 5.0 seconds.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        super().__init__(**kwargs)

        params = params or BaseOpenAILLMService.InputParams()

        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "service_tier": params.service_tier,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
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
        """Create an AsyncOpenAI client instance.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for the API.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.
            **kwargs: Additional client configuration arguments.

        Returns:
            Configured AsyncOpenAI client instance.
        """
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
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI service supports metrics generation.
        """
        return True

    async def get_chat_completions(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API with optional timeout and retry.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Async stream of chat completion chunks.
        """
        params = self.build_chat_completion_params(params_from_context)

        if self._retry_on_timeout:
            try:
                chunks = await asyncio.wait_for(
                    self._client.chat.completions.create(**params), timeout=self._retry_timeout_secs
                )
                return chunks
            except (APITimeoutError, asyncio.TimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying chat completion due to timeout")
                chunks = await self._client.chat.completions.create(**params)
                return chunks
        else:
            chunks = await self._client.chat.completions.create(**params)
            return chunks

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for chat completion request.

        Subclasses can override this to customize parameters for different providers.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
            "service_tier": self._settings["service_tier"],
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings["extra"])
        return params

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        if isinstance(context, LLMContext):
            adapter = self.get_llm_adapter()
            params: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(context)
            messages = params["messages"]
        else:
            messages = context.messages

        # LLM completion
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
        )

        return response.choices[0].message.content

    async def _stream_chat_completions_specific_context(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(
            f"{self}: Generating chat from LLM-specific context {context.get_messages_for_logging()}"
        )

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                # Avoid .getvalue() which makes a full copy of BytesIO
                raw_bytes = message["data"].read()
                encoded_image = base64.b64encode(raw_bytes).decode("utf-8")
                text = message.get("content", "")
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                # Explicit cleanup
                del message["data"]
                del message["mime_type"]

        params = OpenAILLMInvocationParams(
            messages=messages, tools=context.tools, tool_choice=context.tool_choice
        )
        chunks = await self.get_chat_completions(params)

        return chunks

    async def _stream_chat_completions_universal_context(
        self, context: LLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating chat from universal context {adapter.get_messages_for_logging(context)}"
        )

        params: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(context)
        chunks = await self.get_chat_completions(params)

        return chunks

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        # Generate chat completions using either OpenAILLMContext or universal LLMContext
        chunk_stream = await (
            self._stream_chat_completions_specific_context(context)
            if isinstance(context, OpenAILLMContext)
            else self._stream_chat_completions_universal_context(context)
        )

        async for chunk in chunk_stream:
            if chunk.usage:
                cached_tokens = (
                    chunk.usage.prompt_tokens_details.cached_tokens
                    if chunk.usage.prompt_tokens_details
                    else None
                )
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cache_read_input_tokens=cached_tokens,
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

            # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
            # we need to get LLMTextFrame for the transcript
            elif hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio.get(
                "transcript"
            ):
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.audio["transcript"]))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                arguments = json.loads(arguments)
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles OpenAILLMContextFrame, LLMContextFrame, LLMMessagesFrame,
        and LLMUpdateSettingsFrame to trigger LLM completions and manage
        settings.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            # Handle OpenAI-specific context frames
            context = frame.context
        elif isinstance(frame, LLMContextFrame):
            # Handle universal (LLM-agnostic) LLM context frames
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            # NOTE: LLMMessagesFrame is deprecated, so we don't support the newer universal
            # LLMContext with it
            context = OpenAILLMContext.from_messages(frame.messages)
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
