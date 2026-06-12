#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hopper LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class HopperLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for HopperLLMService."""

    pass


class HopperLLMService(OpenAILLMService):
    """A service for interacting with Hopper's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Hopper's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    Hopper serves open-source models optimized for low time to first token,
    targeted at voice agents.
    """

    # Hopper doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = HopperLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.withhopper.com/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Hopper LLM service.

        Args:
            api_key: The API key for accessing Hopper's API.
            base_url: The base URL for Hopper API. Defaults to "https://api.withhopper.com/v1".
            settings: Runtime-updatable settings. Defaults the model to
                "Qwen/Qwen3.6-35B-A3B".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="Qwen/Qwen3.6-35B-A3B")

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Hopper API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Hopper's API.
        """
        logger.debug(f"Creating Hopper client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
