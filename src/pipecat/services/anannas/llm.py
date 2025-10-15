#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anannas AI LLM service implementation.

This module provides an OpenAI-compatible interface for interacting with Anannas AI's
unified model gateway, extending the base OpenAI LLM service functionality.
"""

from typing import Optional

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class AnannasLLMService(OpenAILLMService):
    """A service for interacting with Anannas AI's unified model gateway.

    Provides access to 500+ models from OpenAI, Anthropic, Mistral, Gemini, DeepSeek,
    and other providers through a single OpenAI-compatible interface with built-in
    observability and routing capabilities.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.anannas.ai/v1",
        **kwargs,
    ):
        """Initialize the Anannas AI LLM service.

        Args:
            api_key: The API key for accessing Anannas AI's API. If None, will attempt
                to read from ANANNAS_API_KEY environment variable.
            model: The model identifier to use. Supports any model available through
                Anannas AI (e.g., "gpt-4o", "claude-3-5-sonnet-20241022",
                "deepseek-chat"). Defaults to "gpt-4o".
            base_url: The base URL for Anannas AI API. Defaults to "https://api.anannas.ai/v1".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an Anannas AI API client.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured Anannas AI API client instance.
        """
        logger.debug(f"Creating Anannas AI client with base URL {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

