#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax LLM service implementation.

MiniMax exposes an OpenAI-compatible chat completions endpoint, but its
reasoning models (e.g. ``MiniMax-M2.7``) emit their internal reasoning inline
inside the streaming ``delta.content`` field, wrapped in ``<think>...</think>``
tags. Without filtering, that reasoning text is aggregated into sentences and
forwarded to TTS as if it were the model's answer.

This service extends :class:`OpenAILLMService` to detect a leading
``<think>`` block in the stream and strip it before the base OpenAI loop
sees it. Reasoning content is re-emitted as ``LLMThought*Frame`` objects so
observers and logging can still see it.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import (
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ThinkTagState(StrEnum):
    DETECTING = "detecting"
    IN_THOUGHT = "in_thought"
    CONTENT = "content"


@dataclass
class MiniMaxLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for MiniMaxLLMService."""

    pass


class MiniMaxLLMService(OpenAILLMService):
    """A service for interacting with MiniMax's OpenAI-compatible chat API.

    This service extends :class:`OpenAILLMService` to handle MiniMax's reasoning
    models, which emit their internal reasoning inline in the streaming
    ``delta.content`` field, wrapped in ``<think>...</think>`` tags. The leading
    think block is detected, stripped from the visible content stream, and
    re-emitted as ``LLMThought*Frame`` objects so observers and logging can
    still see the reasoning without it being spoken by TTS.

    Non-reasoning MiniMax models (or models that don't begin with ``<think>``)
    pass through unchanged after a brief one-time detection window.

    Reasoning content is emitted as ``LLMThought*Frame`` objects, keeping it
    accessible to observers and logging without sending it to TTS.
    """

    Settings = MiniMaxLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the MiniMaxLLMService.

        Args:
            api_key: MiniMax API key for authentication.
            base_url: The base URL for the MiniMax chat completions endpoint.
                Defaults to the global endpoint. Use
                ``https://api.minimaxi.chat/v1`` for mainland China or
                ``https://api-uw.minimax.io/v1`` for US-West.
            model: The model identifier to use. Defaults to ``MiniMax-M2.7``.

                .. deprecated:: 0.0.105
                    Use ``settings=MiniMaxLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside
                deprecated parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="MiniMax-M2.7")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        self._reset_think_state()

    def _reset_think_state(self):
        """Reset per-response think-tag detection state."""
        self._think_state = _ThinkTagState.DETECTING
        self._think_buffer = ""

    async def _filter_thinking_content(self, text: str) -> str | None:
        """Filter a leading ``<think>`` block from streamed content.

        Uses a three-state machine that matches the streaming-chunk-safe pattern
        in :class:`pipecat.services.nvidia.llm.NvidiaLLMService`: the opening
        and closing tags may arrive in separate chunks, and the answer may
        share a chunk with the closing tag.

        - ``detecting``: Buffers the start of the stream to check for
          ``<think>``. Non-reasoning chunks transition to ``content`` after the
          buffer grows past the open tag length.
        - ``in_thought``: Inside a think block; buffered text is emitted as
          ``LLMThoughtTextFrame`` until ``</think>`` is found.
        - ``content``: Pass-through.

        Args:
            text: Incremental delta content from the stream.

        Returns:
            The visible content to forward to the base OpenAI processing loop,
            or ``None`` if nothing should be forwarded yet (we're still
            buffering or all the content was reasoning).
        """
        if self._think_state == _ThinkTagState.CONTENT:
            return text

        self._think_buffer += text

        if self._think_state == _ThinkTagState.DETECTING:
            if len(self._think_buffer) < len(_THINK_OPEN):
                if _THINK_OPEN.startswith(self._think_buffer):
                    return None
                self._think_state = _ThinkTagState.CONTENT
                passthrough = self._think_buffer
                self._think_buffer = ""
                return passthrough

            if self._think_buffer.startswith(_THINK_OPEN):
                self._think_state = _ThinkTagState.IN_THOUGHT
                await self.push_frame(LLMThoughtStartFrame())
                self._think_buffer = self._think_buffer[len(_THINK_OPEN) :]
            else:
                self._think_state = _ThinkTagState.CONTENT
                passthrough = self._think_buffer
                self._think_buffer = ""
                return passthrough

        if self._think_state == _ThinkTagState.IN_THOUGHT:
            idx = self._think_buffer.find(_THINK_CLOSE)
            if idx != -1:
                thought = self._think_buffer[:idx]
                if thought:
                    await self.push_frame(LLMThoughtTextFrame(text=thought))
                await self.push_frame(LLMThoughtEndFrame())
                remainder = self._think_buffer[idx + len(_THINK_CLOSE) :]
                self._think_buffer = ""
                self._think_state = _ThinkTagState.CONTENT
                return remainder or None
            # Hold back the trailing chars that could be a partial closing tag.
            safe_end = len(self._think_buffer) - len(_THINK_CLOSE) + 1
            if safe_end > 0:
                await self.push_frame(LLMThoughtTextFrame(text=self._think_buffer[:safe_end]))
                self._think_buffer = self._think_buffer[safe_end:]
        return None

    async def _flush_think_state(self):
        """Flush buffered state at normal stream completion.

        Emits any buffered trailing thought text, closes an open thought block,
        and forwards any buffered pre-content text that was held while deciding
        whether the stream began with ``<think>``.
        """
        if self._think_state == _ThinkTagState.IN_THOUGHT:
            if self._think_buffer:
                await self.push_frame(LLMThoughtTextFrame(text=self._think_buffer))
            await self.push_frame(LLMThoughtEndFrame())
        elif self._think_state == _ThinkTagState.DETECTING and self._think_buffer:
            await super()._push_llm_text(self._think_buffer)

        self._think_buffer = ""
        self._think_state = _ThinkTagState.CONTENT

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the chat completion stream to strip leading ``<think>`` blocks.

        Args:
            context: The LLM context for the completion request.

        Returns:
            An async iterator of chat completion chunks where leading
            ``<think>`` content has been removed from ``delta.content`` and
            re-emitted as ``LLMThought*Frame`` side effects.
        """
        stream = await super().get_chat_completions(context)
        return self._handle_thinking_content(stream)

    async def _handle_thinking_content(
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Strip ``<think>`` blocks from ``delta.content`` across the stream.

        Every chunk is still yielded so the base streaming loop can process
        metadata such as token usage, model name, and tool calls. Stream
        cleanup is owned by the base OpenAI processing loop
        (:meth:`BaseOpenAILLMService._process_context`), which wraps the stream
        in its own closing context manager.

        Args:
            stream: The original chat completion stream.

        Yields:
            Chat completion chunks with leading ``<think>`` content removed
            from ``delta.content`` before they reach the base OpenAI loop.
        """
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    delta.content = await self._filter_thinking_content(delta.content)
            yield chunk

        await self._flush_think_state()

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM, resetting think-tag state first.

        Args:
            context: The context to process, containing messages and other
                information needed for the LLM interaction.
        """
        self._reset_think_state()
        await super()._process_context(context)
