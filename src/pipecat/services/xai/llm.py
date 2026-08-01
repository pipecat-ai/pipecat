#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok LLM service implementation using OpenAI-compatible interface.

This module provides a service for interacting with Grok's API through an
OpenAI-compatible interface, including specialized token usage tracking
and context aggregation functionality.
"""

from dataclasses import dataclass

from loguru import logger

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import (
    OpenAILLMService,
)


@dataclass
class GrokLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for GrokLLMService."""

    pass


class GrokLLMService(OpenAILLMService):
    """A service for interacting with Grok's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Grok's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    Includes specialized token usage tracking that accumulates metrics during
    processing and reports final totals.
    """

    Settings = GrokLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the GrokLLMService with API key and model.

        Args:
            api_key: The API key for accessing Grok's API.
            base_url: The base URL for Grok API. Defaults to "https://api.x.ai/v1".
            model: The model identifier to use. Defaults to "grok-4.20-non-reasoning".

                .. deprecated:: 0.0.105
                    Use ``settings=GrokLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="grok-4.20-non-reasoning",
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)
        # Grok repeats a cumulative usage snapshot on every streamed chunk, so
        # the latest one holds the totals for the whole completion.
        self._token_usage: LLMTokenUsage | None = None

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Grok API endpoint.

        Args:
            api_key: The API key to use. If None, uses instance default.
            base_url: The base URL to use. If None, uses instance default.
            **kwargs: Additional arguments passed to client creation.

        Returns:
            The configured client instance for Grok API.
        """
        logger.debug(f"Creating Grok client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM, reporting usage once per completion.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        self._token_usage = None

        try:
            await super()._process_context(context)
        finally:
            # Only the base implementation emits the metrics; report through it
            # even if the response is interrupted or cancelled mid-stream.
            if self._token_usage:
                await super().start_llm_usage_metrics(self._token_usage)
                self._token_usage = None

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Hold the latest usage snapshot rather than reporting it.

        The inherited streaming loop calls this for every chunk carrying usage.
        Holding the snapshot here suppresses that per-chunk reporting, leaving
        :meth:`_process_context` to report the final one when the completion ends.

        Args:
            tokens: Cumulative token usage for the completion so far.
        """
        self._token_usage = tokens
