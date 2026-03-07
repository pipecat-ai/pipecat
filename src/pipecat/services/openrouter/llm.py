#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter LLM service implementation.

This module provides an OpenAI-compatible interface for interacting with OpenRouter's API,
extending the base OpenAI LLM service functionality.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class OpenRouterLLMSettings(OpenAILLMSettings):
    """Settings for OpenRouterLLMService."""

    pass


class OpenRouterLLMService(OpenAILLMService):
    """A service for interacting with OpenRouter's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to OpenRouter's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    _settings: OpenRouterLLMSettings

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        settings: Optional[OpenRouterLLMSettings] = None,
        **kwargs,
    ):
        """Initialize the OpenRouter LLM service.

        Args:
            api_key: The API key for accessing OpenRouter's API. If None, will attempt
                to read from environment variables.
            model: The model identifier to use. Defaults to "openai/gpt-4o-2024-11-20".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

            base_url: The base URL for OpenRouter API. Defaults to "https://openrouter.ai/api/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = OpenRouterLLMSettings(model="openai/gpt-4o-2024-11-20")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", OpenRouterLLMSettings, "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenRouter API client.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured OpenRouter API client instance.
        """
        logger.debug(f"Creating OpenRouter client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(self, params_from_context: Dict[str, Any]) -> Dict[str, Any]:
        """Builds chat parameters, handling model-specific constraints.

        Args:
            params_from_context: Parameters from the LLM context.

        Returns:
            Transformed parameters ready for the API call.
        """
        params = super().build_chat_completion_params(params_from_context)
        if "gemini" in self._settings.model.lower():
            messages = params.get("messages", [])
            if not messages:
                return params
            transformed_messages = []
            system_message_seen = False
            for msg in messages:
                if msg.get("role") == "system":
                    if not system_message_seen:
                        transformed_messages.append(msg)
                        system_message_seen = True
                    else:
                        new_msg = msg.copy()
                        new_msg["role"] = "user"
                        transformed_messages.append(new_msg)
                else:
                    transformed_messages.append(msg)
            params["messages"] = transformed_messages

        return params
