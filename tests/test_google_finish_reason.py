#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for finish_reason handling in GoogleLLMService."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    Part,
)
from loguru import logger

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService


def _chunk(*, parts=None, finish_reason=None) -> GenerateContentResponse:
    """Build a single streamed chunk holding one candidate."""
    content = Content(role="model", parts=parts) if parts is not None else None
    return GenerateContentResponse(
        candidates=[Candidate(content=content, finish_reason=finish_reason)]
    )


@contextmanager
def _captured_warnings():
    """Collect warning-level log messages emitted within the block."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


async def _stream(*chunks):
    """Run a context through the service against a canned stream.

    Returns:
        The pushed frames and any warnings logged while streaming.
    """
    service = GoogleLLMService(api_key="test-key")
    frames = []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    async def fake_stream(context):
        async def generator():
            for chunk in chunks:
                yield chunk

        return generator()

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_stream_content", fake_stream),
        _captured_warnings() as warnings,
    ):
        await service._process_context(LLMContext())

    return frames, warnings


@pytest.mark.asyncio
async def test_response_is_bracketed_when_no_text_is_generated():
    """An empty response still opens and closes, so aggregators aren't left waiting."""
    frames, warnings = await _stream(_chunk(parts=[], finish_reason=FinishReason.STOP))

    assert isinstance(frames[0], LLMFullResponseStartFrame)
    assert isinstance(frames[-1], LLMFullResponseEndFrame)
    assert warnings == []


@pytest.mark.asyncio
async def test_normal_response_logs_no_warning():
    """A STOP-terminated response with text is passed through untouched."""
    frames, warnings = await _stream(
        _chunk(parts=[Part(text="Hello there.")]),
        _chunk(parts=[], finish_reason=FinishReason.STOP),
    )

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Hello there."]
    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    [
        FinishReason.SAFETY,
        FinishReason.PROHIBITED_CONTENT,
        FinishReason.RECITATION,
        FinishReason.MALFORMED_FUNCTION_CALL,
        FinishReason.OTHER,
    ],
)
async def test_incomplete_response_logs_the_reason(finish_reason):
    """A curtailed response names why, rather than ending as a silent empty turn."""
    frames, warnings = await _stream(_chunk(parts=None, finish_reason=finish_reason))

    assert len(warnings) == 1
    assert finish_reason.name in warnings[0]
    assert isinstance(frames[-1], LLMFullResponseEndFrame)


@pytest.mark.asyncio
async def test_truncated_response_is_passed_through_with_a_warning():
    """Hitting the output token limit is a warning: the partial text is still usable."""
    frames, warnings = await _stream(
        _chunk(parts=[Part(text="Sure, here's the ")]),
        _chunk(parts=[], finish_reason=FinishReason.MAX_TOKENS),
    )

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Sure, here's the "]
    assert len(warnings) == 1
    assert "MAX_TOKENS" in warnings[0]


@pytest.mark.asyncio
async def test_chunks_without_finish_reason_log_no_warning():
    """Intermediate chunks carry no finish reason and must not be reported."""
    frames, warnings = await _stream(
        _chunk(parts=[Part(text="Partial")]),
        _chunk(parts=[Part(text=" text.")]),
    )

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Partial", " text."]
    assert warnings == []


async def _run_inference(*, parts=None, finish_reason=None):
    """Run a one-shot inference against a canned, complete response.

    Returns:
        The returned text and any warnings logged during the call.
    """
    service = GoogleLLMService(api_key="test-key")

    async def generate_content(**kwargs):
        return _chunk(parts=parts, finish_reason=finish_reason)

    with (
        patch.object(service._client.aio.models, "generate_content", generate_content),
        _captured_warnings() as warnings,
    ):
        result = await service.run_inference(LLMContext())

    return result, warnings


@pytest.mark.asyncio
async def test_run_inference_normal_stop_logs_no_warning():
    """A one-shot reply that ended normally is returned without a warning."""
    result, warnings = await _run_inference(
        parts=[Part(text="Hello there.")], finish_reason=FinishReason.STOP
    )

    assert result == "Hello there."
    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", [FinishReason.SAFETY, FinishReason.PROHIBITED_CONTENT])
async def test_run_inference_withheld_reply_logs_the_reason(finish_reason):
    """A withheld one-shot reply returns None and logs why."""
    result, warnings = await _run_inference(parts=None, finish_reason=finish_reason)

    assert result is None
    assert len(warnings) == 1
    assert finish_reason.name in warnings[0]


@pytest.mark.asyncio
async def test_run_inference_truncated_reply_is_returned_with_a_warning():
    """A reply cut off at the output token limit is still returned, with a warning."""
    result, warnings = await _run_inference(
        parts=[Part(text="The conversation so far cove")], finish_reason=FinishReason.MAX_TOKENS
    )

    assert result == "The conversation so far cove"
    assert len(warnings) == 1
    assert "MAX_TOKENS" in warnings[0]
