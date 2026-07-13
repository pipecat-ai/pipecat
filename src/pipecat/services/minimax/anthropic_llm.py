#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax LLM service implementation using the Anthropic-compatible interface."""

from dataclasses import dataclass

from anthropic import AsyncAnthropic
from loguru import logger

from pipecat.services.anthropic.llm import AnthropicLLMService


@dataclass
class MiniMaxAnthropicLLMSettings(AnthropicLLMService.Settings):
    """Settings for MiniMaxAnthropicLLMService."""

    pass


class MiniMaxAnthropicLLMService(AnthropicLLMService):
    """A service for MiniMax's Anthropic-compatible API."""

    Settings = MiniMaxAnthropicLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/anthropic",
        model: str | None = None,
        settings: Settings | None = None,
        client=None,
        **kwargs,
    ):
        """Initialize the MiniMax Anthropic-compatible LLM service.

        Args:
            api_key: The API key for accessing MiniMax's API.
            base_url: The Anthropic-compatible base URL. Defaults to the global endpoint
                ``https://api.minimax.io/anthropic``. Use
                ``https://api.minimaxi.com/anthropic`` for Mainland China.
            model: The model identifier to use. Defaults to ``MiniMax-M3``. The
                ``MiniMax-M2.7`` model is also supported.

                .. deprecated:: 0.0.105
                    Use ``settings=MiniMaxAnthropicLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            client: Optional preconfigured Anthropic-compatible client.
            **kwargs: Additional keyword arguments passed to AnthropicLLMService.
        """
        default_settings = self.Settings(model="MiniMax-M3")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        client = client or AsyncAnthropic(api_key=api_key, base_url=base_url)
        super().__init__(
            api_key=api_key,
            settings=default_settings,
            client=client,
            **kwargs,
        )
        logger.info(
            f"Initialized MiniMax Anthropic-compatible LLM service with model: "
            f"{self._settings.model}"
        )
