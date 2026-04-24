#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anthropic AI service integration for Pipecat.

This module provides LLM services and context management for Anthropic's Claude models,
including support for function calling, vision, and prompt caching features.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    AnthropicLLMInvocationParams,
)
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMEnablePromptCachingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven, assert_given, is_given
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    from anthropic import NOT_GIVEN, APITimeoutError, AsyncAnthropic
    from anthropic import NotGiven as AnthropicNotGiven
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`.")
    raise Exception(f"Missing module: {e}")


class AnthropicThinkingConfig(BaseModel):
    """Configuration for extended thinking.

    Parameters:
        type: Type of thinking mode (currently only "enabled" or "disabled").
        budget_tokens: Maximum number of tokens for thinking.
            With today's models, the minimum is 1024.
            Currently required when type is "enabled", not allowed when "disabled".
    """

    # Why `| str` here? To not break compatibility in case Anthropic adds
    # more types in the future.
    type: Literal["enabled", "disabled"] | str

    # No client-side validation on budget_tokens — we let the server
    # enforce the rules so we stay forward-compatible if they change.
    budget_tokens: int | None = None


@dataclass
class AnthropicLLMSettings(LLMSettings):
    """Settings for AnthropicLLMService.

    Parameters:
        enable_prompt_caching: Whether to enable prompt caching.
        thinking: Extended thinking configuration.
    """

    enable_prompt_caching: bool | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)
    # Override inherited LLMSettings fields to also accept anthropic's NotGiven
    # sentinel. The service stores anthropic's NOT_GIVEN in these fields so
    # they can be passed through unchanged to the AsyncAnthropic client.
    temperature: float | None | _NotGiven | AnthropicNotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    top_k: int | None | _NotGiven | AnthropicNotGiven = field(default_factory=lambda: _NOT_GIVEN)
    top_p: float | None | _NotGiven | AnthropicNotGiven = field(default_factory=lambda: _NOT_GIVEN)
    thinking: Union["AnthropicLLMService.ThinkingConfig", _NotGiven, AnthropicNotGiven] = field(
        default_factory=lambda: _NOT_GIVEN
    )

    @classmethod
    def from_mapping(cls, settings):
        """Convert a plain dict to settings, coercing thinking dicts.

        For backward compatibility, a ``thinking`` value that is a plain dict
        is converted to a :class:`AnthropicLLMService.ThinkingConfig`.
        """
        instance = super().from_mapping(settings)
        if is_given(instance.thinking) and isinstance(instance.thinking, dict):
            instance.thinking = AnthropicLLMService.ThinkingConfig(**instance.thinking)
        return instance


class AnthropicLLMService(LLMService):
    """LLM service for Anthropic's Claude models.

    Provides inference capabilities with Claude models including support for
    function calling, vision processing, streaming responses, and prompt caching.
    Can use custom clients like AsyncAnthropicBedrock and AsyncAnthropicVertex.
    """

    Settings = AnthropicLLMSettings
    _settings: Settings

    # Overriding the default adapter to use the Anthropic one.
    adapter_class = AnthropicLLMAdapter

    # Backward compatibility: ThinkingConfig used to be defined inline here.
    ThinkingConfig = AnthropicThinkingConfig

    class InputParams(BaseModel):
        """Input parameters for Anthropic model inference.

        .. deprecated:: 0.0.105
            Use ``AnthropicLLMService.Settings`` instead. Pass settings directly via the
            ``settings`` parameter of :class:`AnthropicLLMService`.

        Parameters:
            enable_prompt_caching: Whether to enable the prompt caching feature.
            max_tokens: Maximum tokens to generate. Must be at least 1.
            temperature: Sampling temperature between 0.0 and 1.0.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter between 0.0 and 1.0.
            thinking: Extended thinking configuration.
                Enabling extended thinking causes the model to spend more time "thinking" before responding.
                It also causes this service to emit LLMThinking*Frames during response generation.
                Extended thinking is disabled by default.
            extra: Additional parameters to pass to the API.
        """

        enable_prompt_caching: bool | None = None
        max_tokens: int | None = Field(default_factory=lambda: 4096, ge=1)
        temperature: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        top_k: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        top_p: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        thinking: Optional["AnthropicLLMService.ThinkingConfig"] = Field(
            default_factory=lambda: NOT_GIVEN
        )
        extra: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        client=None,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        **kwargs,
    ):
        """Initialize the Anthropic LLM service.

        Args:
            api_key: Anthropic API key for authentication.
            model: Model name to use.

                .. deprecated:: 0.0.105
                    Use ``settings=AnthropicLLMService.Settings(model=...)`` instead.

            params: Optional model parameters for inference.

                .. deprecated:: 0.0.105
                    Use ``settings=AnthropicLLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings for this service.  When both
                deprecated parameters and *settings* are provided, *settings*
                values take precedence.
            client: Optional custom Anthropic client instance.
            retry_timeout_secs: Request timeout in seconds for retry logic.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="claude-sonnet-4-6",
            system_instruction=None,
            max_tokens=4096,
            enable_prompt_caching=False,
            temperature=NOT_GIVEN,
            top_k=NOT_GIVEN,
            top_p=NOT_GIVEN,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            thinking=NOT_GIVEN,
            extra={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.max_tokens = params.max_tokens
                default_settings.temperature = params.temperature
                default_settings.top_k = params.top_k
                default_settings.top_p = params.top_p
                default_settings.thinking = params.thinking
                if isinstance(params.extra, dict):
                    default_settings.extra = params.extra
                if params.enable_prompt_caching is not None:
                    default_settings.enable_prompt_caching = params.enable_prompt_caching

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._client = client or AsyncAnthropic(
            api_key=api_key
        )  # if the client is provided, use it and remove it, otherwise create a new one
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate usage metrics.

        Returns:
            True, as Anthropic provides detailed token usage metrics.
        """
        return True

    async def _create_message_stream(self, api_call, params):
        """Create message stream with optional timeout and retry.

        Args:
            api_call: The Anthropic API method to call.
            params: Parameters for the API call.

        Returns:
            Async stream of message events.
        """
        if self._retry_on_timeout:
            try:
                response = await asyncio.wait_for(
                    api_call(**params), timeout=self._retry_timeout_secs
                )
                return response
            except (TimeoutError, APITimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying message creation due to timeout")
                response = await api_call(**params)
                return response
        else:
            response = await api_call(**params)
            return response

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
                overrides the service's default max_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = NOT_GIVEN
        tools = []
        effective_instruction = system_instruction or assert_given(
            self._settings.system_instruction
        )
        adapter: AnthropicLLMAdapter = self.get_llm_adapter()
        invocation_params = adapter.get_llm_invocation_params(
            context,
            enable_prompt_caching=assert_given(self._settings.enable_prompt_caching),
            system_instruction=effective_instruction,
        )
        messages = invocation_params["messages"]
        system = invocation_params["system"]
        tools = invocation_params["tools"]

        # Build params using the same method as streaming completions
        params = {
            "model": self._settings.model,
            "max_tokens": max_tokens if max_tokens is not None else self._settings.max_tokens,
            "stream": False,
            "temperature": self._settings.temperature,
            "top_k": self._settings.top_k,
            "top_p": self._settings.top_p,
            "messages": messages,
            "system": system,
            "tools": tools,
            "betas": ["interleaved-thinking-2025-05-14"],
        }
        thinking = assert_given(self._settings.thinking)
        if thinking:
            params["thinking"] = thinking.model_dump(exclude_unset=True)

        params.update(self._settings.extra)

        # LLM completion
        response = await self._client.beta.messages.create(**params)

        return next((block.text for block in response.content if hasattr(block, "text")), None)

    def _get_llm_invocation_params(self, context: LLMContext) -> AnthropicLLMInvocationParams:
        adapter: AnthropicLLMAdapter = self.get_llm_adapter()
        params: AnthropicLLMInvocationParams = adapter.get_llm_invocation_params(
            context,
            enable_prompt_caching=assert_given(self._settings.enable_prompt_caching),
            system_instruction=assert_given(self._settings.system_instruction),
        )
        return params

    @traced_llm
    async def _process_context(self, context: LLMContext):
        # Usage tracking. We track the usage reported by Anthropic in prompt_tokens and
        # completion_tokens. We also estimate the completion tokens from output text
        # and use that estimate if we are interrupted, because we almost certainly won't
        # get a complete usage report if the task we're running in is cancelled.
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        use_completion_tokens_estimate = False
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            params_from_context = self._get_llm_invocation_params(context)

            adapter = self.get_llm_adapter()
            messages_for_logging = adapter.get_messages_for_logging(context)
            logger.debug(f"{self}: Generating chat from context {messages_for_logging}")

            await self.start_ttfb_metrics()

            params = {
                "model": self._settings.model,
                "max_tokens": self._settings.max_tokens,
                "stream": True,
                "temperature": self._settings.temperature,
                "top_k": self._settings.top_k,
                "top_p": self._settings.top_p,
            }

            # Add thinking parameter if set
            thinking = assert_given(self._settings.thinking)
            if thinking:
                params["thinking"] = thinking.model_dump(exclude_unset=True)

            # Messages, system, tools
            params.update(params_from_context)

            params.update(self._settings.extra)

            # "Interleaved thinking" needed to allow thinking between sequences
            # of function calls, when extended thinking is enabled.
            # Note that this requires us to use `client.beta`, below.
            params.update({"betas": ["interleaved-thinking-2025-05-14"]})

            response = await self._create_message_stream(self._client.beta.messages.create, params)

            await self.stop_ttfb_metrics()

            # Function calling
            tool_use_block = None
            json_accumulator = ""

            function_calls = []
            async for event in response:
                # Aggregate streaming content, create frames, trigger events

                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        await self._push_llm_text(event.delta.text)
                        completion_tokens_estimate += self._estimate_tokens(event.delta.text)
                    elif hasattr(event.delta, "partial_json") and tool_use_block:
                        json_accumulator += event.delta.partial_json
                        completion_tokens_estimate += self._estimate_tokens(
                            event.delta.partial_json
                        )
                    elif hasattr(event.delta, "thinking"):
                        await self.push_frame(LLMThoughtTextFrame(text=event.delta.thinking))
                    elif hasattr(event.delta, "signature"):
                        await self.push_frame(LLMThoughtEndFrame(signature=event.delta.signature))
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        tool_use_block = event.content_block
                        json_accumulator = ""
                    elif event.content_block.type == "thinking":
                        await self.push_frame(
                            LLMThoughtStartFrame(
                                append_to_context=True,
                                llm=self.get_llm_adapter().id_for_llm_specific_messages,
                            )
                        )
                elif (
                    event.type == "message_delta"
                    and hasattr(event.delta, "stop_reason")
                    and event.delta.stop_reason == "tool_use"
                ):
                    if tool_use_block:
                        args = json.loads(json_accumulator) if json_accumulator else {}
                        function_calls.append(
                            FunctionCallFromLLM(
                                context=context,
                                tool_call_id=tool_use_block.id,
                                function_name=tool_use_block.name,
                                arguments=args,
                            )
                        )

                # Calculate usage. Do this here in its own if statement, because there may be usage
                # data embedded in messages that we do other processing for, above.
                if hasattr(event, "usage"):
                    prompt_tokens += (
                        event.usage.input_tokens if hasattr(event.usage, "input_tokens") else 0
                    )
                    completion_tokens += (
                        event.usage.output_tokens if hasattr(event.usage, "output_tokens") else 0
                    )
                elif hasattr(event, "message") and hasattr(event.message, "usage"):
                    prompt_tokens += (
                        event.message.usage.input_tokens
                        if hasattr(event.message.usage, "input_tokens")
                        else 0
                    )
                    completion_tokens += (
                        event.message.usage.output_tokens
                        if hasattr(event.message.usage, "output_tokens")
                        else 0
                    )
                    cache_creation_input_tokens += (
                        event.message.usage.cache_creation_input_tokens
                        if (
                            hasattr(event.message.usage, "cache_creation_input_tokens")
                            and event.message.usage.cache_creation_input_tokens is not None
                        )
                        else 0
                    )
                    logger.debug(f"Cache creation input tokens: {cache_creation_input_tokens}")
                    cache_read_input_tokens += (
                        event.message.usage.cache_read_input_tokens
                        if (
                            hasattr(event.message.usage, "cache_read_input_tokens")
                            and event.message.usage.cache_read_input_tokens is not None
                        )
                        else 0
                    )
                    logger.debug(f"Cache read input tokens: {cache_read_input_tokens}")
                    total_input_tokens = (
                        prompt_tokens + cache_creation_input_tokens + cache_read_input_tokens
                    )
                    if total_input_tokens >= 1024:
                        if hasattr(
                            context, "turns_above_cache_threshold"
                        ):  # LLMContext doesn't have this attribute
                            context.turns_above_cache_threshold += 1

            await self.run_function_calls(function_calls)

        except asyncio.CancelledError:
            # If we're interrupted, we won't get a complete usage report. So set our flag to use the
            # token estimate. The reraise the exception so all the processors running in this task
            # also get cancelled.
            use_completion_tokens_estimate = True
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            comp_tokens = (
                completion_tokens
                if not use_completion_tokens_estimate
                else completion_tokens_estimate
            )
            await self._report_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and route them appropriately.

        Handles various frame types including context frames, message frames,
        vision frames, and settings updates.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._process_context(frame.context)
        elif isinstance(frame, LLMEnablePromptCachingFrame):
            logger.debug(f"Setting enable prompt caching to: [{frame.enable}]")
            self._settings.enable_prompt_caching = frame.enable
        else:
            await self.push_frame(frame, direction)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(re.split(r"[^\w]+", text)) * 1.3)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
    ):
        if (
            prompt_tokens
            or completion_tokens
            or cache_creation_input_tokens
            or cache_read_input_tokens
        ):
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            await self.start_llm_usage_metrics(tokens)
