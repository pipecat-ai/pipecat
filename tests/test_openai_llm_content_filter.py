#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a reply the provider's content filter cut short is reported.

Azure OpenAI, OpenAI and OpenAI-compatible moderation proxies end a filtered
stream with ``finish_reason: "content_filter"``. The text released before the
verdict is a partial answer, and apps need to know that rather than reading the
turn as a finished one.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import ContentFilterFinishReasonError

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.errors import ErrorCategory


def _text_chunk(text: str):
    """Build a stream chunk carrying a piece of the answer."""
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=None,
            )
        ],
    )


def _finish_chunk(finish_reason: str):
    """Build the chunk that carries the verdict, with no content of its own."""
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=None, finish_reason=finish_reason)],
    )


class _FakeStream:
    """Stands in for the provider's chat completion stream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def close(self):
        pass


def _service(chunks):
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            api_key="test-key", settings=OpenAILLMService.Settings(model="test-model")
        )
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


def _context():
    return LLMContext(messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_a_filtered_reply_is_reported_as_a_content_filter_error():
    """The cut reply keeps its text, and the app is told why it stopped."""
    service = _service([_text_chunk("Here is how"), _finish_chunk("content_filter")])

    with patch.object(FrameProcessor, "push_error", AsyncMock()) as pushed:
        await service._process_context(_context())

    pushed.assert_called_once()
    kwargs = pushed.call_args.kwargs
    assert kwargs["category"] == ErrorCategory.CONTENT_FILTERED
    assert isinstance(kwargs["exception"], ContentFilterFinishReasonError)


@pytest.mark.asyncio
async def test_a_reply_that_finished_is_not_reported_as_filtered():
    """A normal end of turn stays quiet."""
    service = _service([_text_chunk("Done."), _finish_chunk("stop")])

    with patch.object(FrameProcessor, "push_error", AsyncMock()) as pushed:
        await service._process_context(_context())

    pushed.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_raises_on_a_filtered_reply():
    """The one-shot path has no frame pipeline to report through, so it raises."""
    service = _service([])
    service._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Here is how"),
                    finish_reason="content_filter",
                )
            ]
        )
    )

    with pytest.raises(ContentFilterFinishReasonError):
        await service.run_inference(_context())
