#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nebius LLM service implementation using OpenAI-compatible interface."""

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
    """A service for interacting with Nebius's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Nebius's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Nebius doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = NebiusLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.tokenfactory.nebius.com/v1/",
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Nebius LLM service.

        Args:
            api_key: The API key for accessing Nebius's API.
            base_url: The base URL for the Nebius API. Defaults to
                ``"https://api.tokenfactory.nebius.com/v1/"``.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="openai/gpt-oss-120b",
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Nebius API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Nebius's API.
        """
        logger.debug(f"Creating Nebius client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
