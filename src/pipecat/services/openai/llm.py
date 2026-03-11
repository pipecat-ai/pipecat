#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM service implementation with context aggregators."""

import json
from dataclasses import dataclass
from typing import Any, Optional

from openai import NOT_GIVEN

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService


@dataclass
class OpenAIContextAggregatorPair:
    """Pair of OpenAI context aggregators for user and assistant messages.

    .. deprecated:: 0.0.99
        `OpenAIContextAggregatorPair` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.

    Parameters:
        _user: User context aggregator for processing user messages.
        _assistant: Assistant context aggregator for processing assistant messages.
    """

    # Aggregators handle deprecation warnings
    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    """OpenAI LLM service implementation.

    Provides a complete OpenAI LLM service with context aggregation support.
    Uses the BaseOpenAILLMService for core functionality and adds OpenAI-specific
    context aggregator creation.
    """

    Settings = BaseOpenAILLMService.Settings

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        service_tier: Optional[str] = None,
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize OpenAI LLM service.

        Args:
            model: The OpenAI model name to use. Defaults to "gpt-4.1".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMService.Settings(model=...)`` instead.

            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            params: Input parameters for model configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent BaseOpenAILLMService.
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

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # Handle service_tier from deprecated params
        if params is not None and not settings and params.service_tier is not NOT_GIVEN:
            service_tier = service_tier or params.service_tier

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
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

        super().__init__(service_tier=service_tier, settings=default_settings, **kwargs)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create OpenAI-specific context aggregators.

        Creates a pair of context aggregators optimized for OpenAI's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        .. deprecated:: 0.0.99
            `create_context_aggregator()` is deprecated and will be removed in a future version.
            Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
            See `OpenAILLMContext` docstring for migration guide.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        # Aggregators handle deprecation warnings
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)

        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    """OpenAI-specific user context aggregator.

    Handles aggregation of user messages for OpenAI LLM services.
    Inherits all functionality from the base LLMUserContextAggregator.

    .. deprecated:: 0.0.99
        `OpenAIUserContextAggregator` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.
    """

    # Super handles deprecation warning
    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    """OpenAI-specific assistant context aggregator.

    Handles aggregation of assistant messages for OpenAI LLM services,
    with specialized support for OpenAI's function calling format,
    tool usage tracking, and image message handling.

    .. deprecated:: 0.0.99
        `OpenAIAssistantContextAggregator` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.
    """

    # Super handles deprecation warning

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call in progress.

        Adds the function call to the context with an IN_PROGRESS status
        to track ongoing function execution.

        Args:
            frame: Frame containing function call progress information.
        """
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle the result of a function call.

        Updates the context with the function call result, replacing any
        previous IN_PROGRESS status.

        Args:
            frame: Frame containing the function call result.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle a cancelled function call.

        Updates the context to mark the function call as cancelled.

        Args:
            frame: Frame containing the function call cancellation information.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if (
                message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle a user image frame from a function call request.

        Marks the associated function call as completed and adds the image
        to the context for processing.

        Args:
            frame: Frame containing the user image and request context.
        """
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )
