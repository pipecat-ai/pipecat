#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for MiniMax LLM service.

Focused on the streaming ``<think>...</think>`` filter that strips reasoning
content from MiniMax reasoning models (e.g. ``MiniMax-M2.7``) before it
reaches the base OpenAI processing loop and the downstream TTS aggregator.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.services.minimax.llm import MiniMaxLLMService, _ThinkTagState
from pipecat.services.openai.llm import OpenAILLMService


def _build_service() -> MiniMaxLLMService:
    """Construct a MiniMaxLLMService with the OpenAI client patched out."""
    with patch.object(MiniMaxLLMService, "create_client"):
        return MiniMaxLLMService(api_key="test-key")


def _make_chunk(content: str | None):
    """Build a minimal ChatCompletionChunk-like object with one delta."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _drain(text_chunks: list[str], service: MiniMaxLLMService) -> str:
    """Feed text chunks through the filter and concatenate the visible output."""
    out: list[str] = []
    for chunk in text_chunks:
        result = await service._filter_thinking_content(chunk)
        if result is not None:
            out.append(result)
    return "".join(out)


class TestMiniMaxLLMServiceInit:
    def test_default_model(self):
        service = _build_service()
        assert service._settings.model == "MiniMax-M2.7"

    def test_initial_state_is_detecting(self):
        service = _build_service()
        assert service._think_state == _ThinkTagState.DETECTING
        assert service._think_buffer == ""

    def test_settings_override_applies(self):
        with patch.object(MiniMaxLLMService, "create_client"):
            service = MiniMaxLLMService(
                api_key="test-key",
                settings=MiniMaxLLMService.Settings(model="MiniMax-M2.7-highspeed"),
            )
        assert service._settings.model == "MiniMax-M2.7-highspeed"


class TestFilterThinkingContent:
    @pytest.mark.asyncio
    async def test_passthrough_when_no_think_tag(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # Stream starts with regular content — passes through after the
        # one-time DETECTING window resolves.
        out = await _drain(["Hello, how can I help you?"], service)

        assert out == "Hello, how can I help you?"
        assert service._think_state == _ThinkTagState.CONTENT
        service.push_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_complete_think_block_in_single_chunk(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        out = await _drain(
            ["<think>Internal reasoning here.</think>Visible answer."],
            service,
        )

        assert out == "Visible answer."
        assert service._think_state == _ThinkTagState.CONTENT

    @pytest.mark.asyncio
    async def test_handles_open_tag_split_across_chunks(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # Opening "<think>" arrives in two chunks; the DETECTING state must
        # keep buffering until it can decide.
        out = await _drain(
            ["<th", "ink>reasoning</think>answer"],
            service,
        )

        assert out == "answer"

    @pytest.mark.asyncio
    async def test_handles_close_tag_split_across_chunks(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # Closing "</think>" straddles a chunk boundary — must not split
        # the visible answer prematurely.
        out = await _drain(
            ["<think>reasoning</thi", "nk>answer"],
            service,
        )

        assert out == "answer"

    @pytest.mark.asyncio
    async def test_detecting_prefix_that_is_not_think_tag(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # First chunk starts with "<" but isn't "<think>". Once the buffer
        # outgrows "<think>" without matching, it gets flushed as content.
        out = await _drain(["<other>actual answer"], service)

        assert out == "<other>actual answer"
        assert service._think_state == _ThinkTagState.CONTENT

    @pytest.mark.asyncio
    async def test_emits_thought_frames(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        await _drain(
            ["<think>my reasoning</think>visible"],
            service,
        )

        pushed_frames = [call.args[0] for call in service.push_frame.call_args_list]
        frame_types = [type(f) for f in pushed_frames]

        assert LLMThoughtStartFrame in frame_types
        assert LLMThoughtEndFrame in frame_types
        thought_texts = [f.text for f in pushed_frames if isinstance(f, LLMThoughtTextFrame)]
        assert "".join(thought_texts) == "my reasoning"


class TestFlushThinkState:
    @pytest.mark.asyncio
    async def test_flush_pushes_dangling_detection_buffer_as_content(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # The flush invokes ``super()._push_llm_text(...)``, which resolves
        # via the MRO and bypasses instance attributes — patch on the parent
        # class so the mock is reachable.
        with patch.object(OpenAILLMService, "_push_llm_text", new_callable=AsyncMock) as mock_push:
            # Whole response is the prefix "<th" and the stream ends — this
            # wasn't a think tag, so it must still surface to the user.
            result = await service._filter_thinking_content("<th")
            assert result is None
            assert service._think_state == _ThinkTagState.DETECTING

            await service._flush_think_state()

            mock_push.assert_awaited_once_with("<th")

        assert service._think_state == _ThinkTagState.CONTENT
        assert service._think_buffer == ""

    @pytest.mark.asyncio
    async def test_flush_closes_open_thought_block(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # Open a think block but never close it before the stream ends.
        await _drain(["<think>unfinished reasoning"], service)
        assert service._think_state == _ThinkTagState.IN_THOUGHT

        await service._flush_think_state()

        pushed_frames = [call.args[0] for call in service.push_frame.call_args_list]
        # Last frame pushed should be the end-of-thought marker.
        assert isinstance(pushed_frames[-1], LLMThoughtEndFrame)
        assert service._think_state == _ThinkTagState.CONTENT


class TestResetStateBetweenCalls:
    @pytest.mark.asyncio
    async def test_reset_clears_state_for_next_response(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        # First response: open a think block and leave it open.
        await _drain(["<think>partial"], service)
        assert service._think_state == _ThinkTagState.IN_THOUGHT
        assert service._think_buffer != ""

        # Reset (called at the start of every _process_context).
        service._reset_think_state()

        assert service._think_state == _ThinkTagState.DETECTING
        assert service._think_buffer == ""


class TestStreamWrapping:
    @pytest.mark.asyncio
    async def test_handle_thinking_content_filters_chunks(self):
        service = _build_service()
        service.push_frame = AsyncMock()

        async def fake_stream():
            for content in ["<think>hidden</think>", "visible answer"]:
                yield _make_chunk(content)

        out_chunks = []
        async for chunk in service._handle_thinking_content(fake_stream()):
            out_chunks.append(chunk.choices[0].delta.content)

        # Every chunk is still yielded (so the base loop can read metadata),
        # but reasoning content is dropped from delta.content.
        assert out_chunks == [None, "visible answer"]
