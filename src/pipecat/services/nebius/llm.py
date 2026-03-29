#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nebius Token Factory LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class NebiusLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for NebiusLLMService."""

    pass


class NebiusLLMService(OpenAILLMService):
    """A service for interacting with Nebius Token Factory's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Nebius Token Factory's API endpoint
    while maintaining full compatibility with OpenAI's interface and functionality.

    Nebius Token Factory provides access to open-source models including Meta Llama,
    Qwen, and DeepSeek variants through an OpenAI-compatible REST API.

    Set the ``NEBIUS_API_KEY`` environment variable or pass ``api_key`` directly.

    Example::

        service = NebiusLLMService(
            api_key="your-nebius-api-key",
            settings=NebiusLLMService.Settings(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
        )
    """

    Settings = NebiusLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.tokenfactory.nebius.com/v1/",
        model: Optional[str] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Nebius Token Factory LLM service.

        Args:
            api_key: The API key for accessing Nebius Token Factory's API.
            base_url: The base URL for the Nebius API. Defaults to
                ``"https://api.tokenfactory.nebius.com/v1/"``.
            model: The model identifier to use. Defaults to
                ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``.

                .. deprecated:: 0.0.109
                    Use ``settings=NebiusLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="meta-llama/Meta-Llama-3.1-8B-Instruct")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Nebius Token Factory API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Nebius Token Factory's API.
        """
        logger.debug(f"Creating Nebius client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
