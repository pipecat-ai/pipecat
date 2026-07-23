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
        # Kimi K2.5 streams visible content immediately, keeping time-to-first-word low.
        default_settings = self.Settings(
            model="moonshotai/Kimi-K2.5",
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        # Counters for accumulating token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cache_read_input_tokens = 0
        self._reasoning_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

    def _reset_token_usage(self):
        """Reset accumulated token counters at the start of each LLM call."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cache_read_input_tokens = 0
        self._reasoning_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = True

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        Overrides the parent implementation to handle Baseten's cumulative token
        reporting, collapsing the per-chunk counts into a single total reported
        once at the end of processing.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        self._reset_token_usage()

        # Wrap in try/finally so accumulated metrics are still reported if the
        # response is interrupted or cancelled mid-stream.
        try:
            await super()._process_context(context)
        finally:
            self._is_processing = False
            if self._prompt_tokens > 0 or self._completion_tokens > 0:
                tokens = LLMTokenUsage(
                    prompt_tokens=self._prompt_tokens,
                    completion_tokens=self._completion_tokens,
                    total_tokens=self._prompt_tokens + self._completion_tokens,
                    cache_read_input_tokens=self._cache_read_input_tokens or None,
                    reasoning_tokens=self._reasoning_tokens or None,
                )
                await super().start_llm_usage_metrics(tokens)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Accumulate token usage metrics during processing.

        Baseten reports cumulative usage on every streamed chunk rather than
        once at the end, so each update supersedes the previous one instead of
        adding to it.

        Args:
            tokens: Token usage information to accumulate.
        """
        if not self._is_processing:
            return

        # Prompt and cache-read counts are constant across chunks; take the first.
        if not self._has_reported_prompt_tokens and tokens.prompt_tokens > 0:
            self._prompt_tokens = tokens.prompt_tokens
            self._cache_read_input_tokens = tokens.cache_read_input_tokens or 0
            self._has_reported_prompt_tokens = True

        # Completion and reasoning counts grow as the response streams.
        if tokens.completion_tokens > self._completion_tokens:
            self._completion_tokens = tokens.completion_tokens
        reasoning_tokens = tokens.reasoning_tokens or 0
        if reasoning_tokens > self._reasoning_tokens:
            self._reasoning_tokens = reasoning_tokens

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
