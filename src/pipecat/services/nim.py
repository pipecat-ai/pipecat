#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json


from dataclasses import dataclass

from loguru import logger

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.openai import OpenAILLMService
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)


class NimAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def _push_aggregation(self):
        if not (
            self._aggregation or self._function_call_result or self._pending_image_frame_message
        ):
            return

        run_llm = False

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            # "content": "",  # empty content here will break nim
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
                    # Only run the LLM if there are no more function calls in progress.
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

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.exception(f"Error processing frame: {e}")

@dataclass
class NimContextAggregatorPair:
    _user: "OpenAIUserContextAggregator"
    _assistant: "NimAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "NimAssistantContextAggregator":
        return self._assistant

class NimLLMService(OpenAILLMService):
    """A service for interacting with NVIDIA's NIM (NVIDIA Inference Microservice) API.

    This service extends OpenAILLMService to work with NVIDIA's NIM API while maintaining
    compatibility with the OpenAI-style interface. It specifically handles the difference
    in token usage reporting between NIM (incremental) and OpenAI (final summary).

    Args:
        api_key (str): The API key for accessing NVIDIA's NIM API
        base_url (str, optional): The base URL for NIM API. Defaults to "https://integrate.api.nvidia.com/v1"
        model (str, optional): The model identifier to use. Defaults to "nvidia/llama-3.1-nemotron-70b-instruct"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/llama-3.1-nemotron-70b-instruct",
        **kwargs,
    ):
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

        This method intercepts the incremental token updates from NVIDIA's API
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
    ) -> NimContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = NimAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return NimContextAggregatorPair(_user=user, _assistant=assistant)
