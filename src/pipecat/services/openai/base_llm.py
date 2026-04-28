#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base LLM service implementation for services that use the AsyncOpenAI client."""

import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
)
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven
from pipecat.utils.tracing.service_decorators import traced_llm


@dataclass
class OpenAILLMSettings(LLMSettings):
    """Settings for BaseOpenAILLMService.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
    """

    # Override inherited LLMSettings fields to also accept openai's NotGiven
    # sentinel. The service stores openai's NOT_GIVEN in these fields so they
    # can be passed through unchanged to the AsyncOpenAI client.
    frequency_penalty: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    presence_penalty: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    seed: int | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    temperature: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    top_p: float | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    max_tokens: int | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    max_completion_tokens: int | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class BaseOpenAILLMService(LLMService[OpenAILLMAdapter]):
    """Base class for all services that use the AsyncOpenAI client.

    This service consumes LLMContextFrame frames, which contain a reference to
    an LLMContext object. The context defines what is sent to the LLM for
    completion, including user, assistant, and system messages, as well as tool
    choices and function call configurations.
    """

    Settings = OpenAILLMSettings
    _settings: Settings

    supports_developer_role: bool = True
    """Whether this service's API supports the "developer" message role.

    OpenAI's native API supports it, but some OpenAI-compatible services
    (e.g. Cerebras) do not. Subclasses that don't support it should set
    this to ``False``, which causes the adapter to convert "developer"
    messages to "user" messages before sending them to the API.
    """

    class InputParams(BaseModel):
        """Input parameters for OpenAI model configuration.

        .. deprecated:: 0.0.105
            Use ``settings=BaseOpenAILLMService.Settings(...)`` instead of
            ``params=InputParams(...)``.

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

        frequency_penalty: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        presence_penalty: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        seed: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignored right now.
        top_k: int | None = Field(default=None, ge=0)
        top_p: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        max_tokens: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        max_completion_tokens: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        service_tier: str | None = Field(default_factory=lambda: NOT_GIVEN)
        extra: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Mapping[str, str] | None = None,
        service_tier: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        **kwargs,
    ):
        """Initialize the BaseOpenAILLMService.

        Args:
            model: The OpenAI model name to use (e.g., "gpt-4.1", "gpt-4o").

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(model=...)`` instead.

            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            params: Input parameters for model configuration and behavior.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            retry_timeout_secs: Request timeout in seconds. Defaults to 5.0 seconds.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=NOT_GIVEN,
            presence_penalty=NOT_GIVEN,
            seed=NOT_GIVEN,
            temperature=NOT_GIVEN,
            top_p=NOT_GIVEN,
            top_k=None,
            max_tokens=NOT_GIVEN,
            max_completion_tokens=NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )

        # 2. Apply direct init arg overrides (no warnings in base class)
        if model is not None:
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None and not settings:
            default_settings.frequency_penalty = params.frequency_penalty
            default_settings.presence_penalty = params.presence_penalty
            default_settings.seed = params.seed
            default_settings.temperature = params.temperature
            default_settings.top_p = params.top_p
            default_settings.max_tokens = params.max_tokens
            default_settings.max_completion_tokens = params.max_completion_tokens
            if isinstance(params.extra, dict):
                default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )
        self._service_tier = service_tier
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        self._full_model_name: str = ""
        self._client = self.create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            **kwargs,
        )

        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

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

    def set_full_model_name(self, full_model_name: str):
        """Set the full AI model name.

        Args:
            full_model_name: The full name of the AI model to use.
        """
        self._full_model_name = full_model_name

    def get_full_model_name(self):
        """Get the current full model name.

        Returns:
            The full name of the AI model being used.
        """
        return self._full_model_name

    async def get_chat_completions(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API with optional timeout and retry.

        Args:
            context: Context to use for the chat completion.
                Contains messages, tools, and tool choice.

        Returns:
            Async stream of chat completion chunks.
        """
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating chat from context {adapter.get_messages_for_logging(context)}"
        )

        params_from_context: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(
            context,
            system_instruction=self._settings.system_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )

        params = self.build_chat_completion_params(params_from_context)

        if self._retry_on_timeout:
            try:
                chunks = await asyncio.wait_for(
                    self._client.chat.completions.create(**params), timeout=self._retry_timeout_secs
                )
                return chunks
            except (TimeoutError, APITimeoutError):
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
            "model": self._settings.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "seed": self._settings.seed,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
            "max_completion_tokens": self._settings.max_completion_tokens,
            "service_tier": self._service_tier if self._service_tier is not None else NOT_GIVEN,
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings.extra)

        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens/max_completion_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        effective_instruction = system_instruction or self._settings.system_instruction
        adapter = self.get_llm_adapter()
        invocation_params: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(
            context,
            system_instruction=effective_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )

        # Build params using the same method as streaming completions
        params = self.build_chat_completion_params(invocation_params)

        # Override for non-streaming
        params["stream"] = False
        params.pop("stream_options", None)

        # Override max_tokens if provided
        if max_tokens is not None:
            # Use max_completion_tokens for newer models, fallback to max_tokens
            if "max_completion_tokens" in params:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

        # LLM completion
        response = await self._client.chat.completions.create(**params)

        return response.choices[0].message.content

    @traced_llm
    async def _process_context(self, context: LLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        # Generate chat completions from LLMContext
        chunk_stream = await self.get_chat_completions(context)

        # Ensure stream and its async iterator are closed on cancellation/exception
        # to prevent socket leaks and uvloop crashes. Closing the iterator first
        # cascades cleanup through nested async generators (httpx/httpcore internals),
        # preventing uvloop's broken asyncgen finalizer from firing on Python 3.12+
        # (MagicStack/uvloop#699).
        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                # Close the iterator first to cascade cleanup through
                # nested async generators (httpx/httpcore internals).
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                # Then close the stream to release HTTP resources.
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        async with _closing(chunk_stream) as chunk_iter:
            async for chunk in chunk_iter:
                if chunk.usage:
                    cached_tokens = (
                        chunk.usage.prompt_tokens_details.cached_tokens
                        if chunk.usage.prompt_tokens_details
                        else None
                    )
                    reasoning_tokens = (
                        chunk.usage.completion_tokens_details.reasoning_tokens
                        if chunk.usage.completion_tokens_details
                        else None
                    )
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cache_read_input_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.model and self.get_full_model_name() != chunk.model:
                    self.set_full_model_name(chunk.model)

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
                        arguments_list.append(arguments or "{}")
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
                    await self._push_llm_text(chunk.choices[0].delta.content)

                # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
                # we need to get LLMTextFrame for the transcript
                elif (
                    hasattr(chunk.choices[0].delta, "audio")
                    and chunk.choices[0].delta.audio
                    and chunk.choices[0].delta.audio.get("transcript")
                ):
                    await self.push_frame(LLMTextFrame(chunk.choices[0].delta.audio["transcript"]))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Failed to parse function call arguments: {arguments}")
                    continue
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

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
