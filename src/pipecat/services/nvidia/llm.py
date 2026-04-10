#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#

"""NVIDIA NIM API service implementation.

This module provides a service for interacting with NVIDIA's NIM (NVIDIA Inference
Microservice) API while maintaining compatibility with the OpenAI-style interface.

Refer to the NVIDIA NIM LLM API documentation for available models and usage:
https://docs.api.nvidia.com/nim/reference/llm-apis
"""

from dataclasses import dataclass
from typing import AsyncIterator, Optional

from loguru import logger
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import (
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass
class NvidiaLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for NvidiaLLMService."""

    pass


class NvidiaLLMService(OpenAILLMService):
    """A service for interacting with NVIDIA's NIM (NVIDIA Inference Microservice) API.

    This service extends OpenAILLMService to work with NVIDIA's NIM API while
    maintaining compatibility with the OpenAI-style interface. It handles:

    - Incremental token usage reporting (NIM sends per-chunk counts instead
      of a final summary)
    - Automatic detection and filtering of reasoning tokens from models that
      emit ``<think>``/``</think>`` tags in content (e.g. DeepSeek-R1, some nemotron models)
    - Extraction of ``reasoning_content`` from the streaming delta for models
      with API-level reasoning separation (e.g. Nemotron Nano models)

    Reasoning content is emitted as ``LLMThought*Frame`` objects, keeping it
    accessible to observers and logging without sending it to TTS.
    """

    Settings = NvidiaLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the NvidiaLLMService.

        Args:
            api_key: NVIDIA API key for authentication. Required when using the
                cloud endpoint. Not needed for local NIM deployments.
            base_url: The base URL for NIM API. Defaults to NVIDIA's cloud endpoint.
                For local deployments, pass the local address (e.g. ``http://localhost:8000/v1``).
            model: The model identifier to use. Defaults to
                "nvidia/nemotron-3-nano-30b-a3b".

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="nvidia/nemotron-3-nano-30b-a3b")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        if "api.nvidia.com" in base_url and not api_key:
            logger.warning(
                "NvidiaLLMService: Using the cloud endpoint but no API key was provided. "
                "An API key is required for the cloud endpoint. "
                "Set base_url to your local NIM endpoint for local deployments."
            )

        # Counters for accumulating token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False

    def _reset_response_state(self):
        """Reset per-response state at the start of each LLM call.

        Resets token accumulation counters, thinking-tag detection state,
        and reasoning-content field tracking.
        """
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = True

        # <think> tag detection: "detecting" → "in_thought" | "content"
        self._think_tag_state = "detecting"
        self._think_tag_buffer = ""

        # reasoning_content field tracking
        self._has_reasoning_field = False

    async def _push_llm_text(self, text: str):
        """Push LLM text, auto-detecting and filtering ``<think>`` tags.

        Uses a three-state machine to handle reasoning tokens in content:

        - ``detecting``: Buffers the first few chars to check for ``<think>``.
        - ``in_thought``: Inside a think block; emits ``LLMThoughtTextFrame``
          until ``</think>`` is found.
        - ``content``: Normal content; direct passthrough to base class.

        Non-reasoning models transition from ``detecting`` to ``content``
        on the first chunk with zero buffering overhead after that.

        Args:
            text: The text content from the LLM to push.
        """
        if self._think_tag_state == "content":
            await super()._push_llm_text(text)
            return

        self._think_tag_buffer += text

        if self._think_tag_state == "detecting":
            if len(self._think_tag_buffer) < len(_THINK_OPEN):
                if _THINK_OPEN.startswith(self._think_tag_buffer):
                    return
                self._think_tag_state = "content"
                await super()._push_llm_text(self._think_tag_buffer)
                self._think_tag_buffer = ""
                return

            if self._think_tag_buffer.startswith(_THINK_OPEN):
                self._think_tag_state = "in_thought"
                await self.push_frame(LLMThoughtStartFrame())
                self._think_tag_buffer = self._think_tag_buffer[len(_THINK_OPEN) :]
            else:
                self._think_tag_state = "content"
                await super()._push_llm_text(self._think_tag_buffer)
                self._think_tag_buffer = ""
                return

        if self._think_tag_state == "in_thought":
            idx = self._think_tag_buffer.find(_THINK_CLOSE)
            if idx != -1:
                thought = self._think_tag_buffer[:idx]
                if thought:
                    await self.push_frame(LLMThoughtTextFrame(text=thought))
                await self.push_frame(LLMThoughtEndFrame())
                remainder = self._think_tag_buffer[idx + len(_THINK_CLOSE) :]
                self._think_tag_buffer = ""
                self._think_tag_state = "content"
                if remainder:
                    await super()._push_llm_text(remainder)
            else:
                safe_end = len(self._think_tag_buffer) - len(_THINK_CLOSE) + 1
                if safe_end > 0:
                    await self.push_frame(
                        LLMThoughtTextFrame(text=self._think_tag_buffer[:safe_end])
                    )
                    self._think_tag_buffer = self._think_tag_buffer[safe_end:]

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the chat completion stream to handle ``reasoning_content``.

        Models with API-level reasoning separation (e.g. Nemotron Nano)
        include a ``reasoning_content`` field on the streaming delta. This
        wrapper extracts those chunks and emits them as ``LLMThought*Frame``
        objects, keeping them out of the normal content path.

        Args:
            context: The LLM context for the completion request.

        Returns:
            An async iterator of chat completion chunks where
            ``reasoning_content`` has been emitted as ``LLMThought*Frame``
            side effects.
        """
        stream = await super().get_chat_completions(context)
        return self._handle_reasoning_content(stream)

    async def _handle_reasoning_content(
        self, stream: AsyncStream[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle ``reasoning_content`` from a chat completion chunk stream.

        Inspects each chunk for a ``reasoning_content`` field on the delta and
        emits ``LLMThoughtStartFrame`` / ``LLMThoughtTextFrame`` /
        ``LLMThoughtEndFrame`` as side effects. Every chunk (including
        reasoning-only ones) is still yielded so the base streaming loop
        can process metadata such as token usage and model name.

        Notes:
            Stream cleanup is owned by the base OpenAI processing loop
            (``BaseOpenAILLMService._process_context``), which wraps the stream
            in its own closing context manager.

        Args:
            stream: The original chat completion stream.

        Yields:
            All chat completion chunks, unchanged.
        """
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                rc = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if rc:
                    if not self._has_reasoning_field:
                        self._has_reasoning_field = True
                        await self.push_frame(LLMThoughtStartFrame())
                    await self.push_frame(LLMThoughtTextFrame(text=rc))
                elif self._has_reasoning_field and chunk.choices[0].delta.content:
                    await self.push_frame(LLMThoughtEndFrame())
                    self._has_reasoning_field = False
            yield chunk

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        Delegates to the base OpenAI streaming loop while adding
        NVIDIA-specific behavior:

        - ``reasoning_content`` is intercepted via the
          ``get_chat_completions`` stream wrapper and emitted as
          ``LLMThought*Frame`` objects.
        - ``<think>`` tag detection is handled by the ``_push_llm_text``
          override for models that embed reasoning in content.
        - Incremental token counts are accumulated and reported as final
          totals.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        self._reset_response_state()

        # Wrap in try/finally to guarantee accumulated token metrics are
        # reported and _is_processing is cleared even on cancellation.
        try:
            await super()._process_context(context)

            # Flush any pending think-tag state (normal completion only;
            # CancelledError skips this block).
            if self._think_tag_state == "in_thought":
                if self._think_tag_buffer:
                    await self.push_frame(LLMThoughtTextFrame(text=self._think_tag_buffer))
                await self.push_frame(LLMThoughtEndFrame())
            elif self._think_tag_buffer:
                await super()._push_llm_text(self._think_tag_buffer)

            if self._has_reasoning_field:
                await self.push_frame(LLMThoughtEndFrame())
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
