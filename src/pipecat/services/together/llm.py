#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together.ai LLM service implementation using OpenAI-compatible interface."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class TogetherLLMService(OpenAILLMService):
    """A service for interacting with Together.ai's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Together.ai's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        **kwargs,
    ):
        """Initialize Together.ai LLM service.

        Args:
            api_key: The API key for accessing Together.ai's API.
            base_url: The base URL for Together.ai API. Defaults to "https://api.together.xyz/v1".
            model: The model identifier to use. Defaults to "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Together.ai API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Together.ai's API.
        """
        logger.debug(f"Creating Together.ai client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
