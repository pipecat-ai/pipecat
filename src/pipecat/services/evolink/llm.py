#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""EvoLink LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class EvoLinkLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for EvoLinkLLMService."""

    pass


class EvoLinkLLMService(OpenAILLMService):
    """A service for interacting with EvoLink's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to EvoLink's direct API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    supports_developer_role = False

    Settings = EvoLinkLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://direct.evolink.ai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize EvoLink LLM service.

        Args:
            api_key: The API key for accessing EvoLink's API.
            base_url: The base URL for EvoLink API. Defaults to
                "https://direct.evolink.ai/v1".
            model: The model identifier to use. Defaults to "gpt-5.2".

                .. deprecated:: 0.0.105
                    Use ``settings=EvoLinkLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="gpt-5.2")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for EvoLink API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for EvoLink's API.
        """
        logger.debug(f"Creating EvoLink client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
