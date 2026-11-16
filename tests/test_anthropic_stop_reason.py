#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for stop_reason handling in AnthropicLLMService."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from loguru import logger

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _usage():
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        output_tokens_details=None,
    )


def _text(text):
    """Build a streamed text delta."""
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text))


def _stop(stop_reason, category=None):
    """Build the message_delta event that ends a stream."""
    details = SimpleNamespace(type="refusal", category=category) if category else None
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason, stop_details=details),
        usage=_usage(),
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


async def _stream(*events):
    """Run a context through the service against a canned stream.

    Returns:
        The pushed frames and any warnings logged while streaming.
    """
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5-5")
    )
    frames = []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    async def fake_stream(api_call, params):
        async def generator():
            for event in events:
                yield event

        return generator()

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_create_message_stream", fake_stream),
        _captured_warnings() as warnings,
    ):
        await service._process_context(LLMContext())

    return frames, warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "stop_sequence"])
async def test_normal_stop_logs_no_warning(stop_reason):
    """A response that ended the way the model meant it to is passed through untouched."""
    frames, warnings = await _stream(_text("Hello there."), _stop(stop_reason))

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Hello there."]
    assert warnings == []


@pytest.mark.asyncio
async def test_refusal_logs_the_reason_and_category():
    """A refused turn names why, rather than ending as a silent empty turn."""
    frames, warnings = await _stream(_stop("refusal", category="reasoning_extraction"))

    assert len(warnings) == 1
    assert "refusal (reasoning_extraction)" in warnings[0]
    assert isinstance(frames[0], LLMFullResponseStartFrame)
    assert isinstance(frames[-1], LLMFullResponseEndFrame)


@pytest.mark.asyncio
async def test_refusal_without_a_category_logs_the_reason():
    """Anthropic leaves the category out of a refusal that maps to no named one."""
    _, warnings = await _stream(_stop("refusal"))

    assert len(warnings) == 1
    assert warnings[0].endswith("refusal")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_reason", ["max_tokens", "model_context_window_exceeded", "pause_turn", "a_new_reason"]
)
async def test_truncated_response_is_passed_through_with_a_warning(stop_reason):
    """A curtailed response is a warning: the partial text is still usable."""
    frames, warnings = await _stream(_text("Sure, here's the "), _stop(stop_reason))

    assert [f.text for f in frames if isinstance(f, LLMTextFrame)] == ["Sure, here's the "]
    assert len(warnings) == 1
    assert stop_reason in warnings[0]


@pytest.mark.asyncio
async def test_message_delta_without_stop_reason_logs_no_warning():
    """A message_delta that only carries usage must not be reported."""
    _, warnings = await _stream(_text("Partial"), _stop(None))

    assert warnings == []


async def _run_inference(text, stop_reason, category=None):
    """Run a one-shot inference against a canned, complete message.

    Returns:
        The returned text and any warnings logged during the call.
    """
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5-5")
    )
    details = SimpleNamespace(type="refusal", category=category) if category else None
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    message = SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=details)

    async def create(**params):
        return message

    with (
        patch.object(service._client.beta.messages, "create", create),
        _captured_warnings() as warnings,
    ):
        result = await service.run_inference(LLMContext())

    return result, warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "stop_sequence"])
async def test_run_inference_normal_stop_logs_no_warning(stop_reason):
    """A one-shot reply that ended normally is returned without a warning."""
    result, warnings = await _run_inference("Hello there.", stop_reason)

    assert result == "Hello there."
    assert warnings == []


@pytest.mark.asyncio
async def test_run_inference_refusal_logs_the_reason_and_category():
    """A refused one-shot inference returns None and logs why."""
    result, warnings = await _run_inference(None, "refusal", category="reasoning_extraction")

    assert result is None
    assert len(warnings) == 1
    assert "refusal (reasoning_extraction)" in warnings[0]


@pytest.mark.asyncio
async def test_run_inference_truncated_reply_is_returned_with_a_warning():
    """A reply cut off at max_tokens is still returned, with a warning naming the cause."""
    result, warnings = await _run_inference("The conversation so far cove", "max_tokens")

    assert result == "The conversation so far cove"
    assert len(warnings) == 1
    assert "max_tokens" in warnings[0]
