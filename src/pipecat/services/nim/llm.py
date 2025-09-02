#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA NIM API service implementation.

This module provides a service for interacting with NVIDIA's NIM (NVIDIA Inference
Microservice) API while maintaining compatibility with the OpenAI-style interface.
"""

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class NimLLMService(OpenAILLMService):
    """A service for interacting with NVIDIA's NIM (NVIDIA Inference Microservice) API.

    This service extends OpenAILLMService to work with NVIDIA's NIM API while maintaining
    compatibility with the OpenAI-style interface. It specifically handles the difference
    in token usage reporting between NIM (incremental) and OpenAI (final summary).
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/llama-3.1-nemotron-70b-instruct",
        **kwargs,
    ):
        """Initialize the NimLLMService.

        Args:
            api_key: The API key for accessing NVIDIA's NIM API.
            base_url: The base URL for NIM API. Defaults to "https://integrate.api.nvidia.com/v1".
            model: The model identifier to use. Defaults to "nvidia/llama-3.1-nemotron-70b-instruct".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        # Counters for accumulating token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle NVIDIA's
        incremental token reporting style, accumulating the counts and reporting
        them once at the end of processing.

        Args:
            context: The context to process, containing messages and other information
                needed for the LLM interaction.
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

        This method intercepts the incremental token updates from NVIDIA's API
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
