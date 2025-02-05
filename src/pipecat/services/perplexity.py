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


class PerplexityAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Custom assistant context aggregator for Perplexity."""

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
                    function_response = {
                        "role": "assistant",
                        "content": json.dumps(frame.result),
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
                    self._context.add_message(function_response)

                    if properties and properties.run_llm is not None:
                        run_llm = properties.run_llm
                    else:
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

            if properties and properties.on_context_updated is not None:
                await properties.on_context_updated()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")


@dataclass
class PerplexityContextAggregatorPair:
    _user: "OpenAIUserContextAggregator"
    _assistant: "PerplexityAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "PerplexityAssistantContextAggregator":
        return self._assistant


class PerplexityLLMService(OpenAILLMService):
    """A service for interacting with Perplexity's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Perplexity's API endpoint while
    maintaining compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing Perplexity's API.
        base_url (str, optional): The base URL for Perplexity API. Defaults to "https://api.perplexity.ai/v1".
        model (str, optional): The model identifier to use. Defaults to "sonar".
        **kwargs: Additional keyword arguments passed to OpenAILLMService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.perplexity.ai/v1",
        model: str = "sonar",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Perplexity API endpoint."""
        logger.debug(f"Creating Perplexity client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle Perplexity's
        token reporting style, accumulating the counts and reporting them once at the end of processing.

        Args:
            context (OpenAILLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
        """
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = True

        try:
            await super()._process_context(context)
        finally:
            self._is_processing = False
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

        This method intercepts the incremental token updates from Perplexity's API
        and accumulates them instead of passing each update to the metrics system.
        The final accumulated totals are reported at the end of processing.

        Args:
            tokens (LLMTokenUsage): The token usage metrics for the current chunk
                of processing, containing prompt_tokens and completion_tokens counts.
        """
        if not self._is_processing:
            return

        if not self._has_reported_prompt_tokens and tokens.prompt_tokens > 0:
            self._prompt_tokens = tokens.prompt_tokens
            self._has_reported_prompt_tokens = True

        if tokens.completion_tokens > self._completion_tokens:
            self._completion_tokens = tokens.completion_tokens

    def _sanitize_messages(self, messages):
        """Ensure messages alternate between user and assistant roles after system messages."""
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        other_messages = [msg for msg in messages if msg["role"] != "system"]

        if not other_messages:
            return system_messages + [{"role": "user", "content": "Let's begin the conversation."}]

        sanitized_messages = []
        expected_role = "user" if other_messages[0]["role"] == "assistant" else "assistant"

        for msg in other_messages:
            if msg["role"] == expected_role:
                sanitized_messages.append(msg)
                expected_role = "assistant" if expected_role == "user" else "user"
            elif len(sanitized_messages) == 0 or sanitized_messages[-1]["role"] != expected_role:
                sanitized_messages.append(
                    {
                        "role": expected_role,
                        "content": "Continue." if expected_role == "user" else "I understand.",
                    }
                )
                sanitized_messages.append(msg)
                expected_role = "assistant" if msg["role"] == "user" else "user"

        if sanitized_messages and sanitized_messages[-1]["role"] == "assistant":
            sanitized_messages.append({"role": "user", "content": "Please continue."})

        return system_messages + sanitized_messages

    async def get_chat_completions(self, context, messages):
        """Override to sanitize messages before sending to API."""
        sanitized_messages = self._sanitize_messages(messages)
        return await super().get_chat_completions(context, sanitized_messages)

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> PerplexityContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = PerplexityAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return PerplexityContextAggregatorPair(_user=user, _assistant=assistant)
