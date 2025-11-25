#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Perplexity LLM service implementation.

This module provides a service for interacting with Perplexity's API using
an OpenAI-compatible interface. It handles Perplexity's unique token usage
reporting patterns while maintaining compatibility with the Pipecat framework.
"""

from openai import NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class PerplexityLLMService(OpenAILLMService):
    """A service for interacting with Perplexity's API.

    This service extends OpenAILLMService to work with Perplexity's API while maintaining
    compatibility with the OpenAI-style interface. It specifically handles the difference
    in token usage reporting between Perplexity (incremental) and OpenAI (final summary).
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.perplexity.ai",
        model: str = "sonar",
        **kwargs,
    ):
        """Initialize the Perplexity LLM service.

        Args:
            api_key: The API key for accessing Perplexity's API.
            base_url: The base URL for Perplexity's API. Defaults to "https://api.perplexity.ai".
            model: The model identifier to use. Defaults to "sonar".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        # Counters for accumulating token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Perplexity chat completion request.

        Perplexity uses a subset of OpenAI parameters and doesn't support tools.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": params_from_context["messages"],
        }

        # Add OpenAI-compatible parameters if they're set
        if self._settings["frequency_penalty"] is not NOT_GIVEN:
            params["frequency_penalty"] = self._settings["frequency_penalty"]
        if self._settings["presence_penalty"] is not NOT_GIVEN:
            params["presence_penalty"] = self._settings["presence_penalty"]
        if self._settings["temperature"] is not NOT_GIVEN:
            params["temperature"] = self._settings["temperature"]
        if self._settings["top_p"] is not NOT_GIVEN:
            params["top_p"] = self._settings["top_p"]
        if self._settings["max_tokens"] is not NOT_GIVEN:
            params["max_tokens"] = self._settings["max_tokens"]

        return params

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle
        Perplexity's incremental token reporting style, accumulating the counts
        and reporting them once at the end of processing.

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

        Perplexity reports token usage incrementally during streaming,
        unlike OpenAI which provides a final summary. We accumulate the
        counts and report the total at the end of processing.

        Args:
            tokens: Token usage information to accumulate.
        """
        if not self._is_processing:
            return

        # Record prompt tokens the first time we see them
        if not self._has_reported_prompt_tokens and tokens.prompt_tokens > 0:
            self._prompt_tokens = tokens.prompt_tokens
            self._has_reported_prompt_tokens = True

        # Update completion tokens count if it has increased
        if tokens.completion_tokens > self._completion_tokens:
            self._completion_tokens = tokens.completion_tokens
