#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nebius AI Studio LLM service implementation using OpenAI-compatible interface."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class NebiusLLMService(OpenAILLMService):
    """A service for interacting with Nebius AI Studio's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Nebius AI Studio's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.studio.nebius.ai/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        **kwargs,
    ):
        """Initialize Nebius AI Studio LLM service.

        Args:
            api_key: The API key for accessing Nebius AI Studio's API.
            base_url: The base URL for Nebius AI Studio API. Defaults to "https://api.studio.nebius.ai/v1".
            model: The model identifier to use.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Nebius AI Studio API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Nebius AI Studio's API.
        """
        logger.debug(f"Creating Nebius AI Studio client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs) 