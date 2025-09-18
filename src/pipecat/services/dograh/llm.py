#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh LLM Service implementation using OpenAI-compatible interface."""

from typing import Optional

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class DograhLLMService(OpenAILLMService):
    """A unified LLM service using Dograh's API with OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Dograh's unified API endpoint
    while maintaining full compatibility with OpenAI's interface. The actual LLM provider
    (OpenAI, Groq, Google, etc.) is determined by the Dograh backend configuration.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://services.dograh.com/api/v1/llm",
        model: str = "default",
        **kwargs,
    ):
        """Initialize Dograh LLM service.

        Args:
            api_key: The Dograh API key for authentication.
            base_url: The base URL for Dograh API. Defaults to "https://services.dograh.com/api/v1/llm".
            model: The model identifier to use. Options include "default", "fast", "accurate".
                   The actual model used is determined by Dograh backend configuration.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Dograh API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance api_key.
            base_url: Base URL for the API. If None, uses instance base_url.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Dograh's API.
        """
        logger.debug(f"Creating Dograh LLM client with base URL: {base_url or self._base_url}")
        return super().create_client(api_key, base_url, **kwargs)
