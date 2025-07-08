#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Qwen LLM service implementation using OpenAI-compatible interface."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class QwenLLMService(OpenAILLMService):
    """A service for interacting with Alibaba Cloud's Qwen LLM API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Qwen's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        **kwargs,
    ):
        """Initialize the Qwen LLM service.

        Args:
            api_key: The API key for accessing Qwen's API (DashScope API key).
            base_url: Base URL for Qwen API. Defaults to "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".
            model: The model identifier to use. Defaults to "qwen-plus".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        logger.info(f"Initialized Qwen LLM service with model: {model}")

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Qwen API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation.

        Returns:
            An OpenAI-compatible client configured for Qwen's API.
        """
        logger.debug(f"Creating Qwen client with base URL: {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
