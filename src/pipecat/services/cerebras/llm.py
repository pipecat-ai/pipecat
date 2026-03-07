#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cerebras LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class CerebrasLLMSettings(OpenAILLMSettings):
    """Settings for CerebrasLLMService."""

    pass


class CerebrasLLMService(OpenAILLMService):
    """A service for interacting with Cerebras's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Cerebras's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    _settings: CerebrasLLMSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.cerebras.ai/v1",
        model: Optional[str] = None,
        settings: Optional[CerebrasLLMSettings] = None,
        **kwargs,
    ):
        """Initialize the Cerebras LLM service.

        Args:
            api_key: The API key for accessing Cerebras's API.
            base_url: The base URL for Cerebras API. Defaults to "https://api.cerebras.ai/v1".
            model: The model identifier to use. Defaults to "gpt-oss-120b".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = CerebrasLLMSettings(model="gpt-oss-120b")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", CerebrasLLMSettings, "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Cerebras API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance key.
            base_url: The base URL for the API. If None, uses instance URL.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Cerebras API.
        """
        logger.debug(f"Creating Cerebras client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Cerebras chat completion request.

        Cerebras supports a subset of OpenAI parameters, focusing on core
        completion settings without advanced features like frequency/presence penalties.

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
            "seed": self._settings.seed,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_completion_tokens": self._settings.max_completion_tokens,
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
