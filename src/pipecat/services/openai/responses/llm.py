#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Responses API LLM service implementation."""

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from pipecat.adapters.services.open_ai_responses_adapter import (
    OpenAIResponsesLLMAdapter,
    OpenAIResponsesLLMInvocationParams,
)
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven
from pipecat.utils.tracing.service_decorators import traced_llm


@dataclass
class OpenAIResponsesLLMSettings(LLMSettings):
    """Settings for OpenAIResponsesLLMService.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
    """

    max_completion_tokens: int | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)


class OpenAIResponsesLLMService(LLMService):
    """OpenAI Responses API LLM service.

    This service works with the universal LLMContext and LLMContextAggregatorPair.

    Example::

        llm = OpenAIResponsesLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    Settings = OpenAIResponsesLLMSettings
    _settings: Settings

    adapter_class = OpenAIResponsesLLMAdapter

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Optional[Mapping[str, str]] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the OpenAI Responses API LLM service.

        Args:
            model: The OpenAI model name to use. Defaults to "gpt-4.1".
            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            settings: Runtime-updatable settings. When provided alongside
                other parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            temperature=NOT_GIVEN,
            top_p=NOT_GIVEN,
            top_k=None,
            max_tokens=None,
            max_completion_tokens=NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )

        if model is not None:
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )

        self._client = self._create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
        )

        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

    def _create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
    ) -> AsyncOpenAI:
        """Create an AsyncOpenAI client instance.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for the API.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.

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
        """Check if this service can generate processing metrics."""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, LLMContextFrame):
            context = frame.context
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    @traced_llm
    async def _process_context(self, context: LLMContext):
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context "
            f"{adapter.get_messages_for_logging(context)}"
        )

        invocation_params: OpenAIResponsesLLMInvocationParams = adapter.get_llm_invocation_params(
            context, system_instruction=self._settings.system_instruction
        )

        params = self._build_response_params(invocation_params)

        await self.start_ttfb_metrics()

        stream: AsyncStream[ResponseStreamEvent] = await self._client.responses.create(**params)

        # Track function calls across stream events
        function_calls: Dict[str, Dict[str, str]] = {}  # item_id -> {name, call_id, arguments}
        current_arguments: Dict[str, str] = {}  # item_id -> accumulated arguments

        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        async with _closing(stream) as event_iter:
            async for event in event_iter:
                if isinstance(event, ResponseTextDeltaEvent):
                    await self.stop_ttfb_metrics()
                    await self._push_llm_text(event.delta)

                elif isinstance(event, ResponseOutputItemAddedEvent):
                    await self.stop_ttfb_metrics()
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        item_id = getattr(item, "id", "") or ""
                        function_calls[item_id] = {
                            "name": getattr(item, "name", ""),
                            "call_id": getattr(item, "call_id", ""),
                            "arguments": "",
                        }
                        current_arguments[item_id] = ""

                elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                    item_id = event.item_id
                    if item_id in current_arguments:
                        current_arguments[item_id] += event.delta

                elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                    item_id = event.item_id
                    if item_id in function_calls:
                        function_calls[item_id]["arguments"] = event.arguments

                elif isinstance(event, ResponseOutputItemDoneEvent):
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        item_id = getattr(item, "id", "") or ""
                        if item_id in function_calls:
                            function_calls[item_id]["name"] = getattr(item, "name", "")
                            function_calls[item_id]["call_id"] = getattr(item, "call_id", "")
                            function_calls[item_id]["arguments"] = getattr(item, "arguments", "")

                elif isinstance(event, ResponseCompletedEvent):
                    response = event.response
                    usage = getattr(response, "usage", None)
                    if usage:
                        tokens = LLMTokenUsage(
                            prompt_tokens=getattr(usage, "input_tokens", 0),
                            completion_tokens=getattr(usage, "output_tokens", 0),
                            total_tokens=getattr(usage, "total_tokens", 0),
                        )
                        await self.start_llm_usage_metrics(tokens)

                    model = getattr(response, "model", None)
                    if model:
                        self._full_model_name = model

        # Process any function calls
        if function_calls:
            fc_list: List[FunctionCallFromLLM] = []
            for item_id, fc in function_calls.items():
                try:
                    arguments = json.loads(fc["arguments"]) if fc["arguments"] else {}
                except json.JSONDecodeError:
                    logger.warning(
                        f"{self}: Failed to parse function call arguments: {fc['arguments']}"
                    )
                    arguments = {}
                fc_list.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=fc["call_id"],
                        function_name=fc["name"],
                        arguments=arguments,
                    )
                )
            await self.run_function_calls(fc_list)

    def _build_response_params(self, invocation_params: OpenAIResponsesLLMInvocationParams) -> dict:
        """Build parameters for the responses.create() call.

        Args:
            invocation_params: Parameters derived from the LLM context.

        Returns:
            Dictionary of parameters for the Responses API call.
        """
        params: Dict[str, Any] = {
            "model": self._settings.model,
            "stream": True,
            "input": invocation_params["input"],
        }

        # instructions (set by the adapter when input is non-empty)
        if "instructions" in invocation_params:
            params["instructions"] = invocation_params["instructions"]

        # Optional parameters - only include if given
        if isinstance(self._settings.temperature, (int, float)):
            params["temperature"] = self._settings.temperature

        if isinstance(self._settings.top_p, (int, float)):
            params["top_p"] = self._settings.top_p

        if isinstance(self._settings.max_completion_tokens, int):
            params["max_output_tokens"] = self._settings.max_completion_tokens

        # Tools
        tools = invocation_params.get("tools")
        if tools is not None and not isinstance(tools, type(NOT_GIVEN)):
            params["tools"] = tools

        # Extra settings
        params.update(self._settings.extra)

        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Run a one-shot, out-of-band inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        adapter = self.get_llm_adapter()
        effective_instruction = system_instruction or self._settings.system_instruction
        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=effective_instruction
        )

        params = self._build_response_params(invocation_params)

        # Override for non-streaming
        params["stream"] = False

        # Override instructions if caller provided one explicitly
        if system_instruction is not None:
            params["instructions"] = system_instruction

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        response = await self._client.responses.create(**params)

        return response.output_text


__all__ = ["OpenAIResponsesLLMService", "OpenAIResponsesLLMSettings"]
