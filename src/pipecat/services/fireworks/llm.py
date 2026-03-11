#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fireworks AI service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class FireworksLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for FireworksLLMService."""

    pass


class FireworksLLMService(OpenAILLMService):
    """A service for interacting with Fireworks AI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Fireworks' API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = FireworksLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Fireworks LLM service.

        Args:
            api_key: The API key for accessing Fireworks AI.
            model: The model identifier to use. Defaults to "accounts/fireworks/models/firefunction-v2".

                .. deprecated:: 0.0.105
                    Use ``settings=FireworksLLMService.Settings(model=...)`` instead.

            base_url: The base URL for Fireworks API. Defaults to "https://api.fireworks.ai/inference/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="accounts/fireworks/models/firefunction-v2")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Fireworks API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            Configured OpenAI client instance for Fireworks API.
        """
        logger.debug(f"Creating Fireworks client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Fireworks chat completion request.

        Fireworks doesn't support some OpenAI parameters like seed, max_completion_tokens,
        and stream_options.

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
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings.extra)

        # Prepend system instruction if set
        if self._settings.system_instruction:
            messages = params.get("messages", [])
            params["messages"] = [
                {"role": "system", "content": self._settings.system_instruction}
            ] + messages

        return params
