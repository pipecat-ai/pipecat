#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Synthorai LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class SynthoraiLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for SynthoraiLLMService."""

    pass


class SynthoraiLLMService(OpenAILLMService):
    """A service for interacting with Synthorai's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Synthorai's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = SynthoraiLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://synthorai.io/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Synthorai LLM service.

        Args:
            api_key: The API key for accessing Synthorai's API.
            base_url: The base URL for Synthorai API. Defaults to "https://synthorai.io/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(
            model="claude-opus-5",
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Synthorai's API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Synthorai's API.
        """
        logger.debug(f"Creating Synthorai client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
