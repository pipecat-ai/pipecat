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

import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger
from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import (
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.text.tagged_llm_text_parser import (
    TaggedLLMTextParsedOutputChunk,
    TaggedLLMTextParsedOutputChunkKind,
    TaggedLLMTextParser,
)


@dataclass
class NvidiaLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for NvidiaLLMService."""

    pass


class NvidiaLLMService(OpenAILLMService):
    """A service for interacting with NVIDIA's NIM (NVIDIA Inference Microservice) API.

    This service extends OpenAILLMService to work with NVIDIA's NIM API while
    maintaining compatibility with the OpenAI-style interface. It handles:

    - Detection and filtering of leading ``<think>``/``</think>`` content for
      models that emit reasoning inline before visible output (e.g.
      DeepSeek-R1, some nemotron models)
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
        api_key: str | None = None,
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
                "nvidia/nemotron-3.5-super-vl-120b-a12b".

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="nvidia/nemotron-3.5-super-vl-120b-a12b")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Local NIM takes no key, but the OpenAI client needs one: given None it
        # would fall back to OPENAI_API_KEY and send it to NVIDIA.
        super().__init__(
            api_key=api_key or "not-needed",
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

        if "api.nvidia.com" in base_url and not api_key:
            logger.warning(
                "NvidiaLLMService: Using the cloud endpoint but no API key was provided. "
                "An API key is required for the cloud endpoint. "
                "Set base_url to your local NIM endpoint for local deployments."
            )

    def _reset_response_state(self):
        """Reset per-response state at the start of each LLM call.

        Resets leading-think-tag detection state and reasoning-content field
        tracking.
        """
        self._think_parser = TaggedLLMTextParser("<think>", "</think>")

        # reasoning_content field tracking
        self._has_reasoning_field = False

    async def _filter_thinking_content(self, text: str) -> str | None:
        """Filter leading ``<think>`` tags from content and emit thought frames.

        Returns only visible content to the base OpenAI processing loop while
        emitting a leading ``<think>`` block as ``LLMThought*Frame`` side
        effects. Responses without one pass through from the first chunk that
        rules it out.

        Args:
            text: The text content from the LLM to filter.

        Returns:
            The non-reasoning content that should continue through the base
            OpenAI content path, or ``None`` if this chunk should not emit
            normal content.
        """
        visible = ""
        for chunk in self._think_parser.parse_next_chunk(text):
            if chunk.kind == TaggedLLMTextParsedOutputChunkKind.TEXT:
                visible += chunk.text
            else:
                await self._push_thought_chunk(chunk)
        return visible or None

    async def _finalize_reasoning_state(self, *, flush_buffered_text: bool):
        """Finalize buffered reasoning state at stream end.

        Args:
            flush_buffered_text: Whether to forward buffered text that was held
                while deciding whether the stream started with ``<think>``.
                This should be ``True`` on normal completion and ``False``
                when the stream ends early due to interruption or
                cancellation.
        """
        chunks = self._think_parser.flush() if flush_buffered_text else self._think_parser.reset()
        for chunk in chunks:
            if chunk.kind == TaggedLLMTextParsedOutputChunkKind.TEXT:
                await super()._push_llm_text(chunk.text)
            else:
                await self._push_thought_chunk(chunk)

        if self._has_reasoning_field:
            await self.push_frame(LLMThoughtEndFrame())
            self._has_reasoning_field = False

    async def _push_thought_chunk(self, chunk: TaggedLLMTextParsedOutputChunk):
        """Push a piece of a ``<think>`` block as a thought frame."""
        if chunk.kind == TaggedLLMTextParsedOutputChunkKind.THOUGHT_START:
            await self.stop_ttfb_metrics()
            await self.push_frame(LLMThoughtStartFrame())
        elif chunk.kind == TaggedLLMTextParsedOutputChunkKind.THOUGHT_TEXT:
            await self.push_frame(LLMThoughtTextFrame(text=chunk.text))
        elif chunk.kind == TaggedLLMTextParsedOutputChunkKind.THOUGHT_END:
            await self.push_frame(LLMThoughtEndFrame())

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the chat completion stream to handle ``reasoning_content``.

        Models with API-level reasoning separation (e.g. Nemotron Nano)
        include a ``reasoning_content`` field on the streaming delta. This
        wrapper extracts those chunks and emits them as ``LLMThought*Frame``
        objects. It also rewrites streamed ``delta.content`` so leading
        ``<think>`` sections are removed before the base OpenAI loop processes
        visible content.

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
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle ``reasoning_content`` and leading ``<think>`` tags in a chunk stream.

        Inspects each chunk for a ``reasoning_content`` field on the delta and
        emits ``LLMThoughtStartFrame`` / ``LLMThoughtTextFrame`` /
        ``LLMThoughtEndFrame`` as side effects. It also strips ``<think>``
        blocks from ``delta.content`` before yielding the chunk so the base
        OpenAI loop only sees user-facing content. Every chunk is still yielded
        so the base streaming loop can process metadata such as token usage,
        model name, tool calls, and audio transcripts.

        Notes:
            ``BaseOpenAILLMService._process_context()`` closes the wrapper
            iterator returned from ``get_chat_completions()``, but it does not
            close the inner OpenAI stream directly. This wrapper closes that
            inner stream in a ``finally`` block so it is released promptly,
            including if the response is cancelled very early, for example due
            to an interruption right after the request starts.

        Args:
            stream: The original chat completion stream.

        Yields:
            Chat completion chunks with any leading ``<think>`` content removed
            from ``delta.content`` before they reach the base OpenAI loop.
        """
        completed = False
        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    rc = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if rc:
                        if not self._has_reasoning_field:
                            await self.stop_ttfb_metrics()
                            self._has_reasoning_field = True
                            await self.push_frame(LLMThoughtStartFrame())
                        await self.push_frame(LLMThoughtTextFrame(text=rc))
                    elif self._has_reasoning_field and delta.content:
                        await self.push_frame(LLMThoughtEndFrame())
                        self._has_reasoning_field = False

                    if delta.content:
                        delta.content = await self._filter_thinking_content(delta.content)
                yield chunk
            completed = True
        finally:
            try:
                await self._finalize_reasoning_state(
                    flush_buffered_text=completed,
                )
            finally:
                await self._close_inner_stream(stream)

    async def _close_inner_stream(self, stream: AsyncIterator[ChatCompletionChunk]) -> None:
        """Eagerly close the underlying OpenAI streaming response.

        The OpenAI Python SDK exposes ``close()`` on this stream object.
        Closing here complements the base OpenAI cleanup path and keeps
        teardown local to this adapter, including early interruption or
        cancellation cases.
        """
        close = getattr(stream, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "NvidiaLLMService: error while closing underlying chat completion stream"
            )

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM.

        Delegates to the base OpenAI streaming loop, resetting the per-response
        state it needs to emit ``LLMThought*Frame`` objects for
        ``reasoning_content`` and leading ``<think>`` content, which the
        ``get_chat_completions`` stream wrapper intercepts.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        self._reset_response_state()
        await super()._process_context(context)
