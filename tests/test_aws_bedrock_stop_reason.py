#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for stopReason handling in AWSBedrockLLMService."""

from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService


def _text(text):
    """Build a streamed text delta."""
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


def _stop(stop_reason):
    """Build the event that ends a message."""
    return {"messageStop": {"stopReason": stop_reason}}


@contextmanager
def _captured_warnings():
    """Collect warning-level log messages emitted within the block."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _service():
    service = AWSBedrockLLMService(
        aws_access_key="test", aws_secret_key="test", aws_region="us-east-1"
    )
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()

    @asynccontextmanager
    async def create_client(**kwargs):
        yield MagicMock()

    service._aws_session = MagicMock()
    service._aws_session.create_client = create_client
    return service


async def _stream(*events):
    """Run a context through the service against a canned stream.

    Returns:
        The pushed text and any warnings logged while streaming.
    """
    service = _service()
    frames = []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    async def fake_stream(client, request_params):
        async def generator():
            for event in events:
                yield event

        return {"stream": generator()}

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_create_converse_stream", fake_stream),
        _captured_warnings() as warnings,
    ):
        await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    return [f.text for f in frames if isinstance(f, LLMTextFrame)], warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "stop_sequence"])
async def test_normal_stop_logs_no_warning(stop_reason):
    """A response that ended the way the model meant it to is passed through untouched."""
    text, warnings = await _stream(_text("Hello there."), _stop(stop_reason))

    assert text == ["Hello there."]
    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_reason",
    [
        "max_tokens",
        "guardrail_intervened",
        "content_filtered",
        "model_context_window_exceeded",
        "a_new_reason",
    ],
)
async def test_incomplete_response_is_passed_through_with_a_warning(stop_reason):
    """A curtailed response is a warning: the partial text is still usable."""
    text, warnings = await _stream(_text("Sure, here's the "), _stop(stop_reason))

    assert text == ["Sure, here's the "]
    assert len(warnings) == 1
    assert stop_reason in warnings[0]


@pytest.mark.asyncio
async def test_stream_without_a_message_stop_logs_no_warning():
    """Only a messageStop event is reported."""
    _, warnings = await _stream(_text("Partial"))

    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("stop_reason", "warns"), [("end_turn", False), ("max_tokens", True)])
async def test_run_inference_reports_a_truncated_reply(stop_reason, warns):
    """A one-shot reply cut off at the token limit is still returned, with a warning."""
    service = _service()
    client = MagicMock()
    client.converse = AsyncMock(
        return_value={
            "output": {"message": {"content": [{"text": "The conversation so far cove"}]}},
            "stopReason": stop_reason,
        }
    )

    @asynccontextmanager
    async def create_client(**kwargs):
        yield client

    service._aws_session.create_client = create_client

    with _captured_warnings() as warnings:
        result = await service.run_inference(
            LLMContext(messages=[{"role": "user", "content": "Hi"}])
        )

    assert result == "The conversation so far cove"
    assert bool(warnings) is warns
    if warns:
        assert "max_tokens" in warnings[0]
