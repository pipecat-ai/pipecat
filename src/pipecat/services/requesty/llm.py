#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Requesty LLM service implementation.

This module provides an OpenAI-compatible interface for interacting with Requesty's API,
extending the base OpenAI LLM service functionality.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import assert_given


@dataclass
class RequestyLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for RequestyLLMService."""

    pass


class RequestyLLMService(OpenAILLMService):
    """A service for interacting with Requesty's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Requesty's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality. Requesty
    uses the same ``provider/model`` naming convention as OpenRouter (e.g.
    ``openai/gpt-4.1``, ``anthropic/claude-sonnet-4-5``).
    """

    Settings = RequestyLLMSettings
    _settings: Settings
    supports_developer_role = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://router.requesty.ai/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Requesty LLM service.

        Args:
            api_key: The API key for accessing Requesty's API. If None, will attempt
                to read from environment variables.
            model: The model identifier to use. Defaults to "openai/gpt-4.1".

                .. deprecated:: 0.0.105
                    Use ``settings=RequestyLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            base_url: The base URL for Requesty API. Defaults to "https://router.requesty.ai/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="openai/gpt-4.1")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
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
        """Create a Requesty API client.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured Requesty API client instance.
        """
        logger.debug(f"Creating Requesty client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict[str, Any]:
        """Builds chat parameters, handling model-specific constraints.

        Args:
            params_from_context: Parameters from the LLM context.

        Returns:
            Transformed parameters ready for the API call.
        """
        params = super().build_chat_completion_params(params_from_context)
        model = assert_given(self._settings.model)
        if model is not None and "gemini" in model.lower():
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
