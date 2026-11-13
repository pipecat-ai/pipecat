#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for NVIDIA NIM LLM service behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from pipecat.frames.frames import (
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    MetricsFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openai.llm import OpenAILLMService


class _FakeStream:
    """Minimal asynchronous completion stream for service tests."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _reasoning_chunk(*, reasoning_content: str | None, content: str | None):
    """Build an OpenAI-compatible chunk containing NVIDIA reasoning output."""
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    reasoning_content=reasoning_content,
                    content=content,
                    tool_calls=None,
                )
            )
        ],
    )


def _service():
    """Create an NVIDIA service with its API client disabled."""
    with patch.object(NvidiaLLMService, "create_client"):
        return NvidiaLLMService(
            api_key="test-key",
            settings=NvidiaLLMService.Settings(model="test-model"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning_content", "content"),
    [
        pytest.param("Let me think.", None, id="reasoning-content"),
        pytest.param(None, "<think>Let me think.</think>Answer", id="think-tags"),
    ],
)
async def test_reasoning_output_reports_ttfb_before_thought_frames(reasoning_content, content):
    """Reasoning is observable only after its TTFB metric is reported."""
    service = _service()
    pushed = []

    async def capture(frame, *args, **kwargs):
        pushed.append(frame)

    service.push_frame = capture
    service._push_llm_text = AsyncMock()

    stream = _FakeStream([_reasoning_chunk(reasoning_content=reasoning_content, content=content)])
    with (
        patch.object(
            OpenAILLMService,
            "get_chat_completions",
            AsyncMock(return_value=stream),
        ),
        patch.object(
            FrameProcessor,
            "metrics_enabled",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    ttfb_frames = [frame for frame in pushed if isinstance(frame, MetricsFrame)]
    assert len(ttfb_frames) == 1
    assert isinstance(pushed[0], MetricsFrame)
    assert isinstance(pushed[1], LLMThoughtStartFrame)


@pytest.mark.asyncio
async def test_non_reasoning_output_reports_ttfb_without_thought_frames():
    """A normal response reports TTFB without emitting thought frames."""
    service = _service()
    pushed = []

    async def capture(frame, *args, **kwargs):
        pushed.append(frame)

    service.push_frame = capture
    service._push_llm_text = AsyncMock()

    stream = _FakeStream([_reasoning_chunk(reasoning_content=None, content="Answer")])
    with (
        patch.object(
            OpenAILLMService,
            "get_chat_completions",
            AsyncMock(return_value=stream),
        ),
        patch.object(
            FrameProcessor,
            "metrics_enabled",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    ttfb_frames = [frame for frame in pushed if isinstance(frame, MetricsFrame)]
    assert len(ttfb_frames) == 1
    assert not any(
        isinstance(frame, (LLMThoughtStartFrame, LLMThoughtTextFrame, LLMThoughtEndFrame))
        for frame in pushed
    )
    service._push_llm_text.assert_awaited_once_with("Answer")
