#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM service implementation with context aggregators."""

from typing import Optional

from openai import NOT_GIVEN

from pipecat.services.openai.base_llm import BaseOpenAILLMService


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
