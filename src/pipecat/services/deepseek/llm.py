#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DeepSeek LLM service implementation using OpenAI-compatible interface."""

from typing import List

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.llm import OpenAILLMService


class DeepSeekLLMService(OpenAILLMService):
    """A service for interacting with DeepSeek's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to DeepSeek's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        **kwargs,
    ):
        """Initialize the DeepSeek LLM service.

        Args:
            api_key: The API key for accessing DeepSeek's API.
            base_url: The base URL for DeepSeek API. Defaults to "https://api.deepseek.com/v1".
            model: The model identifier to use. Defaults to "deepseek-chat".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for DeepSeek API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for DeepSeek's API.
        """
        logger.debug(f"Creating DeepSeek client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def _build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for DeepSeek chat completion request.

        DeepSeek doesn't support some OpenAI parameters like seed and max_completion_tokens.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings["extra"])
        return params
