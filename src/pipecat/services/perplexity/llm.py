#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Perplexity LLM service implementation.

This module provides a service for interacting with Perplexity's API using
an OpenAI-compatible interface.
"""

from dataclasses import dataclass

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.adapters.services.perplexity_adapter import PerplexityLLMAdapter
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class PerplexityLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for PerplexityLLMService."""

    pass


class PerplexityLLMService(OpenAILLMService):
    """A service for interacting with Perplexity's API.

    This service extends OpenAILLMService to work with Perplexity's API while maintaining
    compatibility with the OpenAI-style interface.
    """

    adapter_class = PerplexityLLMAdapter
    # Perplexity doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = PerplexityLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.perplexity.ai",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Perplexity LLM service.

        Args:
            api_key: The API key for accessing Perplexity's API.
            base_url: The base URL for Perplexity's API. Defaults to "https://api.perplexity.ai".
            model: The model identifier to use. Defaults to "sonar".

                .. deprecated:: 0.0.105
                    Use ``settings=PerplexityLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="sonar")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Perplexity chat completion request.

        Perplexity uses a subset of OpenAI parameters and doesn't support tools.

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
            "messages": params_from_context["messages"],
        }

        # Add OpenAI-compatible parameters if they're set
        if self._settings.frequency_penalty is not None:
            params["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty is not None:
            params["presence_penalty"] = self._settings.presence_penalty
        if self._settings.temperature is not None:
            params["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            params["top_p"] = self._settings.top_p
        if self._settings.max_tokens is not None:
            params["max_tokens"] = self._settings.max_tokens

        return params
