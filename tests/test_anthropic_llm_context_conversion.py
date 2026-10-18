#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AnthropicLLMService's file-message cleanup on error."""

from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest

from pipecat.adapters.base_llm_adapter import LLMContextConversionError
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _make_service() -> AnthropicLLMService:
    service = AnthropicLLMService(api_key="test-key")
    # push_frame is mocked, so pushed error frames go nowhere — but push_error
    # itself stays real so error classification and the usability verdict run.
    service.push_frame = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    return service


async def _context_with_file_message() -> LLMContext:
    context = LLMContext()
    context.add_message({"role": "user", "content": "hello"})
    await context.add_file_frame_message(
        type="bytes", format="application/pdf", file="data:application/pdf;base64,ZmFrZSBwZGY="
    )
    return context


def _api_status_error(error_cls, status_code: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"type": "invalid_request_error", "message": "bad document"}}
    response = httpx.Response(status_code, request=request, json=body)
    return error_cls("bad document", response=response, body=body)


@pytest.mark.asyncio
async def test_anthropic_llm_removes_file_message_on_context_conversion_error():
    """Test that an LLMContextConversionError triggers file-message cleanup.

    An unsupported file MIME type (or corrupt base64) fails during local
    message conversion, before any request reaches Anthropic at all — so it
    never surfaces as an API rejection, but it's just as much evidence the
    file was the problem. Without this, the file stays in context forever,
    since it was never actually sent for Anthropic to reject.
    """
    service = _make_service()
    context = await _context_with_file_message()

    async def raising_create_message_stream(api_call, params):
        raise LLMContextConversionError(ValueError("Unsupported 'file' MIME type"))

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    messages = context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_anthropic_llm_removes_file_message_on_bad_request_error():
    """A 4xx APIStatusError from Anthropic (e.g. a rejected document) triggers cleanup.

    This prevents the context from getting permanently stuck retrying a file
    Anthropic has already rejected.
    """
    service = _make_service()
    context = await _context_with_file_message()

    async def raising_create_message_stream(api_call, params):
        raise _api_status_error(anthropic.BadRequestError, 400)

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    messages = context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
    # The removal changed the next request, so the "permanent" invalid-request
    # verdict must not write the service off.
    assert service.is_usable


@pytest.mark.asyncio
async def test_anthropic_llm_removes_image_message_on_bad_request_error():
    """A rejected image (e.g. bytes that don't match the declared media type)
    gets the same cleanup as a rejected file, so the context doesn't get
    permanently stuck resending it."""
    service = _make_service()
    context = LLMContext()
    context.add_message({"role": "user", "content": "hello"})
    await context.add_image_frame_message(format="image/png", size=(2, 2), image=b"RIFF webp")

    async def raising_create_message_stream(api_call, params):
        raise _api_status_error(anthropic.BadRequestError, 400)

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    messages = context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
    assert service.is_usable


@pytest.mark.asyncio
async def test_anthropic_llm_bad_request_without_a_file_marks_service_unusable():
    """With nothing to repair, an invalid request really will keep recurring,
    so the standard permanent-error verdict stands."""
    service = _make_service()
    context = LLMContext()
    context.add_message({"role": "user", "content": "hello"})

    async def raising_create_message_stream(api_call, params):
        raise _api_status_error(anthropic.BadRequestError, 400)

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    assert len(context.get_messages()) == 1
    assert not service.is_usable


@pytest.mark.asyncio
async def test_anthropic_llm_leaves_context_alone_on_server_error():
    """A 5xx APIStatusError from Anthropic does not trigger file-message cleanup.

    A server-side failure isn't evidence that our request (or its file) was
    bad, so the message should survive to be retried.
    """
    service = _make_service()
    context = await _context_with_file_message()

    async def raising_create_message_stream(api_call, params):
        raise _api_status_error(anthropic.InternalServerError, 500)

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    assert len(context.get_messages()) == 2


@pytest.mark.asyncio
async def test_anthropic_llm_leaves_context_alone_on_rate_limit_error():
    """A 429 RateLimitError from Anthropic does not trigger file-message cleanup.

    A rate limit is a 4xx, but it says nothing about whether the request (or
    its file) was bad — removing the file wouldn't help the next retry
    succeed, and would just discard it for no reason.
    """
    service = _make_service()
    context = await _context_with_file_message()

    async def raising_create_message_stream(api_call, params):
        raise _api_status_error(anthropic.RateLimitError, 429)

    with patch.object(service, "_create_message_stream", raising_create_message_stream):
        await service._process_context(context)

    assert len(context.get_messages()) == 2
