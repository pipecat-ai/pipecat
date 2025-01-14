#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import json
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAILLMService,
    OpenAIUserContextAggregator,
)


class GrokAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Custom assistant context aggregator for Grok that handles empty content requirement."""

    async def _push_aggregation(self):
        if not (
            self._aggregation or self._function_call_result or self._pending_image_frame_message
        ):
            return

        run_llm = False
        properties: Optional[FunctionCallResultProperties] = None

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                properties = frame.properties
                self._function_call_result = None
                if frame.result:
                    # Grok requires an empty content field for function calls
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "content": "",  # Required by Grok
                            "tool_calls": [
                                {
                                    "id": frame.tool_call_id,
                                    "function": {
                                        "name": frame.function_name,
                                        "arguments": json.dumps(frame.arguments),
                                    },
                                    "type": "function",
                                }
                            ],
                        }
                    )
                    self._context.add_message(
                        {
                            "role": "tool",
                            "content": json.dumps(frame.result),
                            "tool_call_id": frame.tool_call_id,
                        }
                    )
                    if properties and properties.run_llm is not None:
                        # If the tool call result has a run_llm property, use it
                        run_llm = properties.run_llm
                    else:
                        # Default behavior is to run the LLM if there are no function calls in progress
                        run_llm = not bool(self._function_calls_in_progress)

            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if self._pending_image_frame_message:
                frame = self._pending_image_frame_message
                self._pending_image_frame_message = None
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text,
                )
                run_llm = True

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            # Emit the on_context_updated callback once the function call result is added to the context
            if properties and properties.on_context_updated is not None:
                await properties.on_context_updated()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")


@dataclass
class GrokContextAggregatorPair:
    _user: "OpenAIUserContextAggregator"
    _assistant: "GrokAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "GrokAssistantContextAggregator":
        return self._assistant


class GrokLLMService(OpenAILLMService):
    """A service for interacting with Grok's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Grok's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing Grok's API
        base_url (str, optional): The base URL for Grok API. Defaults to "https://api.x.ai/v1"
        model (str, optional): The model identifier to use. Defaults to "grok-beta"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-beta",
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
        """Create OpenAI-compatible client for Grok API endpoint."""
        logger.debug(f"Creating Grok client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle Grok's
        incremental token reporting style, accumulating the counts and reporting
        them once at the end of processing.

        Args:
            context (OpenAILLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
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
            tokens (LLMTokenUsage): The token usage metrics for the current chunk
                of processing, containing prompt_tokens and completion_tokens counts.
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

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> GrokContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = GrokAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return GrokContextAggregatorPair(_user=user, _assistant=assistant)
