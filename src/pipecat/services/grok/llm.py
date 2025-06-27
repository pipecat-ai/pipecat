#
# Copyright (c) 2024–2025, Daily
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
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAILLMService,
    OpenAIUserContextAggregator,
)


@dataclass
class GrokContextAggregatorPair:
    """Pair of context aggregators for user and assistant interactions.

    Provides a convenient container for managing both user and assistant
    context aggregators together for Grok LLM interactions.

    Parameters:
        _user: The user context aggregator instance.
        _assistant: The assistant context aggregator instance.
    """

    _user: OpenAIUserContextAggregator
    _assistant: OpenAIAssistantContextAggregator

    def user(self) -> OpenAIUserContextAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> OpenAIAssistantContextAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class GrokLLMService(OpenAILLMService):
    """A service for interacting with Grok's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Grok's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    Includes specialized token usage tracking that accumulates metrics during
    processing and reports final totals.

    Args:
        api_key: The API key for accessing Grok's API.
        base_url: The base URL for Grok API. Defaults to "https://api.x.ai/v1".
        model: The model identifier to use. Defaults to "grok-3-beta".
        **kwargs: Additional keyword arguments passed to OpenAILLMService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-3-beta",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        # Initialize counters for token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

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

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle Grok's
        incremental token reporting style, accumulating the counts and reporting
        them once at the end of processing.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        # Reset all counters and flags at the start of processing
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = True

        try:
            await super()._process_context(context)
        finally:
            self._is_processing = False
            # Report final accumulated token usage at the end of processing
            if self._prompt_tokens > 0 or self._completion_tokens > 0:
                self._total_tokens = self._prompt_tokens + self._completion_tokens
                tokens = LLMTokenUsage(
                    prompt_tokens=self._prompt_tokens,
                    completion_tokens=self._completion_tokens,
                    total_tokens=self._total_tokens,
                )
                await super().start_llm_usage_metrics(tokens)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Accumulate token usage metrics during processing.

        This method intercepts the incremental token updates from Grok's API
        and accumulates them instead of passing each update to the metrics system.
        The final accumulated totals are reported at the end of processing.

        Args:
            tokens: The token usage metrics for the current chunk of processing,
                containing prompt_tokens and completion_tokens counts.
        """
        # Only accumulate metrics during active processing
        if not self._is_processing:
            return

        # Record prompt tokens the first time we see them
        if not self._has_reported_prompt_tokens and tokens.prompt_tokens > 0:
            self._prompt_tokens = tokens.prompt_tokens
            self._has_reported_prompt_tokens = True

        # Update completion tokens count if it has increased
        if tokens.completion_tokens > self._completion_tokens:
            self._completion_tokens = tokens.completion_tokens

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GrokContextAggregatorPair:
        """Create an instance of GrokContextAggregatorPair from an OpenAILLMContext.

        Constructor keyword arguments for both the user and assistant aggregators
        can be provided.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for configuring the user aggregator.
            assistant_params: Parameters for configuring the assistant aggregator.

        Returns:
            GrokContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            GrokContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return GrokContextAggregatorPair(_user=user, _assistant=assistant)
