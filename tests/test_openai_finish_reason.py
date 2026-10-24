#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for finish_reason handling in OpenAI-compatible services."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _chunk(content=None, finish_reason=None):
    """Build a streamed chunk holding one choice."""
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


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
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            api_key="test-key", settings=OpenAILLMService.Settings(model="test-model")
        )
    service._client = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


async def _stream(*chunks):
    """Run a context through the service against a canned stream.

    Returns:
        The pushed text and any warnings logged while streaming.
    """
    service = _service()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    frames = []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    with patch.object(service, "push_frame", capture_frame), _captured_warnings() as warnings:
        await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    return [f.text for f in frames if isinstance(f, LLMTextFrame)], warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "function_call"])
async def test_normal_stop_logs_no_warning(finish_reason):
    """A response that ended the way the model meant it to is passed through untouched."""
    text, warnings = await _stream(_chunk("Hello there."), _chunk(finish_reason=finish_reason))

    assert text == ["Hello there."]
    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "a_new_reason"])
async def test_incomplete_response_is_passed_through_with_a_warning(finish_reason):
    """A curtailed response is a warning: the partial text is still usable."""
    text, warnings = await _stream(_chunk("Sure, here's the "), _chunk(finish_reason=finish_reason))

    assert text == ["Sure, here's the "]
    assert len(warnings) == 1
    assert finish_reason in warnings[0]


@pytest.mark.asyncio
async def test_chunks_without_a_finish_reason_log_no_warning():
    """Only the chunk that carries a finish_reason is reported."""
    _, warnings = await _stream(_chunk("Par"), _chunk("tial"))

    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("finish_reason", "warns"), [("stop", False), ("length", True)])
async def test_run_inference_reports_a_truncated_reply(finish_reason, warns):
    """A one-shot reply cut off at the token limit is still returned, with a warning."""
    service = _service()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The conversation so far cove"),
                finish_reason=finish_reason,
            )
        ]
    )
    service._client.chat.completions.create = AsyncMock(return_value=response)

    with _captured_warnings() as warnings:
        result = await service.run_inference(
            LLMContext(messages=[{"role": "user", "content": "Hi"}])
        )

    assert result == "The conversation so far cove"
    assert bool(warnings) is warns
    if warns:
        assert "length" in warnings[0]
