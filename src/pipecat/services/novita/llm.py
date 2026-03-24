#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Novita AI LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class NovitaLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for NovitaLLMService."""

    pass


class NovitaLLMService(OpenAILLMService):
    """A service for interacting with Novita AI's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Novita AI's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = NovitaLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.novita.ai/openai",
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize Novita AI LLM service.

        Args:
            api_key: The API key for accessing Novita AI's API.
            base_url: The base URL for Novita AI API. Defaults to "https://api.novita.ai/openai".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(
            model="moonshotai/kimi-k2.5",
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
        """Create OpenAI-compatible client for Novita AI API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Novita AI's API.
        """
        logger.debug(f"Creating Novita AI client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
