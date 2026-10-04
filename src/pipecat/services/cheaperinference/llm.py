#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cheaper Inference LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class CheaperInferenceLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for CheaperInferenceLLMService."""

    pass


class CheaperInferenceLLMService(OpenAILLMService):
    """A service for interacting with Cheaper Inference's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Cheaper Inference's API endpoint
    while maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = CheaperInferenceLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.cheaperinference.com/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Cheaper Inference LLM service.

        Args:
            api_key: The API key for accessing Cheaper Inference's API.
            base_url: The base URL for Cheaper Inference API. Defaults to
                "https://api.cheaperinference.com/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(
            model="gpt-5.4-mini",
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
        """Create OpenAI-compatible client for Cheaper Inference API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Cheaper Inference's API.
        """
        logger.debug(f"Creating Cheaper Inference client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
