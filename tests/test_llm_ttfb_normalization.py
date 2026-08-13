#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that LLM services end TTFB at the model's first output.

TTFB is comparable across services only when they all stop it on the same thing:
the first output the model produces, rather than an earlier event that merely
acknowledges the request. These tests pin that boundary for the services whose
streams open with such an event.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService


async def _stop_index(service, patch_stream, chunks) -> int | None:
    """Stream canned chunks and report which one ended TTFB.

    Returns:
        Index of the chunk being processed when TTFB stopped, or None if it
        never stopped.
    """
    state = {"index": -1, "stopped_at": None}

    async def fake_stop_ttfb(**kwargs):
        if state["stopped_at"] is None:
            state["stopped_at"] = state["index"]

    async def generator():
        for index, chunk in enumerate(chunks):
            state["index"] = index
            yield chunk

    async def capture_frame(frame, direction=None):
        pass

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "stop_ttfb_metrics", fake_stop_ttfb),
        patch_stream(service, generator),
    ):
        await service._process_context(LLMContext())

    return state["stopped_at"]


# -- Google -----------------------------------------------------------------


def _google_patch_stream(service, generator):
    """Patch GoogleLLMService's stream with a canned chunk generator."""

    async def fake_stream(context):
        return generator()

    return patch.object(service, "_stream_content", fake_stream)


def _google_usage_chunk() -> GenerateContentResponse:
    """A chunk carrying only usage metadata, with no candidates."""
    return GenerateContentResponse(
        candidates=None,
        usage_metadata=GenerateContentResponseUsageMetadata(prompt_token_count=10),
    )


def _google_text_chunk(text: str) -> GenerateContentResponse:
    """A chunk carrying model output."""
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text=text)]))]
    )


@pytest.mark.asyncio
async def test_google_usage_only_chunk_does_not_stop_ttfb():
    service = GoogleLLMService(api_key="test-key")
    stopped_at = await _stop_index(
        service,
        _google_patch_stream,
        [_google_usage_chunk(), _google_text_chunk("Hello")],
    )
    assert stopped_at == 1


@pytest.mark.asyncio
async def test_google_stops_ttfb_on_first_output_chunk():
    service = GoogleLLMService(api_key="test-key")
    stopped_at = await _stop_index(
        service,
        _google_patch_stream,
        [_google_text_chunk("Hello")],
    )
    assert stopped_at == 0


# -- Anthropic --------------------------------------------------------------


def _anthropic_patch_stream(service, generator):
    """Patch AnthropicLLMService's stream with a canned event generator."""

    async def fake_stream(api_call, params):
        return generator()

    return patch.object(service, "_create_message_stream", fake_stream)


def _message_start() -> SimpleNamespace:
    """The event opening an Anthropic stream, which carries no model output."""
    return SimpleNamespace(type="message_start")


def _content_block_start(block_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start", content_block=SimpleNamespace(type=block_type)
    )


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text))


def _thinking_delta(thinking: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(thinking=thinking))


@pytest.mark.asyncio
async def test_anthropic_message_start_does_not_stop_ttfb():
    service = AnthropicLLMService(api_key="test-key")
    stopped_at = await _stop_index(
        service,
        _anthropic_patch_stream,
        [_message_start(), _content_block_start("text"), _text_delta("Hello")],
    )
    assert stopped_at == 1


@pytest.mark.asyncio
async def test_anthropic_stops_ttfb_on_thinking_before_any_answer_text():
    """Reasoning is output, so it ends TTFB just as answer text would."""
    service = AnthropicLLMService(api_key="test-key")
    stopped_at = await _stop_index(
        service,
        _anthropic_patch_stream,
        [
            _message_start(),
            _content_block_start("thinking"),
            _thinking_delta("Considering..."),
            _content_block_start("text"),
            _text_delta("Hello"),
        ],
    )
    assert stopped_at == 1
