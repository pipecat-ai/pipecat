#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Baseten LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class BasetenLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for BasetenLLMService."""

    pass


class BasetenLLMService(OpenAILLMService):
    """A service for interacting with Baseten's OpenAI-compatible inference API.

    Defaults to Baseten's Model APIs, a serverless endpoint hosting open-weights
    models. To use a dedicated deployment running on your own GPUs, point
    ``base_url`` at it and set ``model`` to the deployment's served model name::

        BasetenLLMService(
            api_key=os.getenv("BASETEN_API_KEY"),
            base_url="https://model-{model_id}.api.baseten.co/environments/production/sync/v1",
            settings=BasetenLLMService.Settings(model="Qwen/Qwen2.5-3B-Instruct"),
        )
    """

    # Baseten's message role enum is system/user/assistant/tool.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = BasetenLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://inference.baseten.co/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Baseten LLM service.

        Args:
            api_key: The API key for accessing Baseten's inference API.
            base_url: The base URL for the Baseten API. Defaults to
                ``"https://inference.baseten.co/v1"``. Set this to a dedicated
                deployment's ``/sync/v1`` URL to use your own hosted model.
            settings: Runtime-updatable settings; values override the built-in defaults.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # Initialize default_settings with hardcoded defaults.
        # Kimi K2.6 streams visible content immediately, keeping time-to-first-word low.
        default_settings = self.Settings(
            model="moonshotai/Kimi-K2.6",
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        # Baseten repeats a cumulative usage snapshot on every streamed chunk, so
        # the latest one holds the totals for the whole completion.
        self._token_usage: LLMTokenUsage | None = None

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

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Baseten API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Baseten's API.
        """
        logger.debug(f"Creating Baseten client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
