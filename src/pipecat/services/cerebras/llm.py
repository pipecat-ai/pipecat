#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cerebras LLM service implementation using OpenAI-compatible interface."""

from typing import List

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class CerebrasLLMService(OpenAILLMService):
    """A service for interacting with Cerebras's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Cerebras's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.cerebras.ai/v1",
        model: str = "llama-3.3-70b",
        **kwargs,
    ):
        """Initialize the Cerebras LLM service.

        Args:
            api_key: The API key for accessing Cerebras's API.
            base_url: The base URL for Cerebras API. Defaults to "https://api.cerebras.ai/v1".
            model: The model identifier to use. Defaults to "llama-3.3-70b".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

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

    def build_chat_completion_params(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> dict:
        """Build parameters for Cerebras chat completion request.

        Cerebras supports a subset of OpenAI parameters, focusing on core
        completion settings without advanced features like frequency/presence penalties.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])
        return params
