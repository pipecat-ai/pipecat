#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AWSBedrockLLMService's file-message cleanup on error."""

from unittest.mock import AsyncMock, patch

import pytest
from botocore.exceptions import ClientError

from pipecat.adapters.base_llm_adapter import LLMContextConversionError
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService


def _make_service() -> AWSBedrockLLMService:
    service = AWSBedrockLLMService(
        aws_access_key="test", aws_secret_key="test", aws_region="us-east-1"
    )
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    return service


async def _context_with_file_message() -> LLMContext:
    context = LLMContext()
    context.add_message({"role": "user", "content": "hello"})
    await context.add_file_frame_message(
        type="bytes", format="application/pdf", file="data:application/pdf;base64,abc123"
    )
    return context


@pytest.mark.asyncio
async def test_aws_bedrock_llm_removes_file_message_on_context_conversion_error():
    """Test that an LLMContextConversionError triggers file-message cleanup.

    An unsupported file MIME type (or corrupt base64) fails during local
    message conversion, before any request reaches Bedrock at all — so it
    never surfaces as an API rejection, but it's just as much evidence the
    file was the problem. Without this, the file stays in context forever,
    since it was never actually sent for Bedrock to reject.
    """
    service = _make_service()
    context = await _context_with_file_message()

    error = LLMContextConversionError(ValueError("Unsupported 'file' MIME type"))
    with patch.object(service, "_get_llm_invocation_params", side_effect=error):
        await service._process_context(context)

    messages = context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_aws_bedrock_llm_removes_file_message_on_client_error():
    """A 4xx ClientError from Bedrock (e.g. a rejected document) triggers cleanup.

    This prevents the context from getting permanently stuck retrying a file
    Bedrock has already rejected (e.g. ValidationException on document name).
    """
    service = _make_service()
    context = await _context_with_file_message()

    error = ClientError(
        {
            "Error": {"Code": "ValidationException", "Message": "bad document name"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "ConverseStream",
    )
    with patch.object(service, "_get_llm_invocation_params", side_effect=error):
        await service._process_context(context)

    messages = context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_aws_bedrock_llm_leaves_context_alone_on_server_error():
    """A 5xx ClientError from Bedrock does not trigger file-message cleanup.

    A server-side failure isn't evidence that our request (or its file) was
    bad, so the message should survive to be retried.
    """
    service = _make_service()
    context = await _context_with_file_message()

    error = ClientError(
        {
            "Error": {"Code": "InternalServerException", "Message": "internal error"},
            "ResponseMetadata": {"HTTPStatusCode": 500},
        },
        "ConverseStream",
    )
    with patch.object(service, "_get_llm_invocation_params", side_effect=error):
        await service._process_context(context)

    assert len(context.get_messages()) == 2
