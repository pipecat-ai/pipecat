#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``LLMFunctionCallDetectedFrame`` and a user-side processor built on it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMFunctionCallDetectedFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.tests.utils import run_test


class SuppressTextAfterFunctionCall(FrameProcessor):
    """What an application would write: drop the text a response streams after a call."""

    def __init__(self):
        super().__init__()
        self._suppressing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._suppressing = False
        elif isinstance(frame, LLMFunctionCallDetectedFrame):
            self._suppressing = True
        elif isinstance(frame, LLMTextFrame) and self._suppressing:
            return
        await self.push_frame(frame, direction)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _chunk(*, content=None, tool_calls=None):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))],
    )


def _tool_call(index=0):
    return SimpleNamespace(
        index=index,
        id=f"call_{index}",
        function=SimpleNamespace(name="lookup", arguments="{}"),
    )


def _service(*responses):
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="test-model"))
    service.get_chat_completions = AsyncMock(side_effect=[_FakeStream(r) for r in responses])
    service.run_function_calls = AsyncMock()
    return service


def _context_frame():
    return LLMContextFrame(LLMContext(messages=[{"role": "user", "content": "Look it up"}]))


def _texts(frames):
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


TOOL_CALL_RESPONSE = [
    _chunk(content="Let me check."),
    _chunk(tool_calls=[_tool_call(0)]),
    _chunk(tool_calls=[_tool_call(1)]),
    _chunk(content="Your order shipped yesterday."),
]


@pytest.mark.asyncio
async def test_frame_is_pushed_once_in_order_with_the_text():
    down, _ = await run_test(_service(TOOL_CALL_RESPONSE), frames_to_send=[_context_frame()])

    kinds = [
        type(f)
        for f in down
        if isinstance(
            f,
            (
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFunctionCallDetectedFrame,
                LLMFullResponseEndFrame,
            ),
        )
    ]
    assert kinds == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFunctionCallDetectedFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_no_frame_without_a_function_call():
    down, _ = await run_test(
        _service([_chunk(content="Hello.")]), frames_to_send=[_context_frame()]
    )

    assert not any(isinstance(f, LLMFunctionCallDetectedFrame) for f in down)


@pytest.mark.asyncio
async def test_user_processor_drops_text_after_the_call():
    down, _ = await run_test(
        Pipeline([_service(TOOL_CALL_RESPONSE), SuppressTextAfterFunctionCall()]),
        frames_to_send=[_context_frame()],
    )

    assert _texts(down) == ["Let me check."]


@pytest.mark.asyncio
async def test_detection_resets_with_the_next_response():
    down, _ = await run_test(
        Pipeline(
            [
                _service(TOOL_CALL_RESPONSE, [_chunk(content="Anything else?")]),
                SuppressTextAfterFunctionCall(),
            ]
        ),
        frames_to_send=[_context_frame(), _context_frame()],
    )

    assert _texts(down) == ["Let me check.", "Anything else?"]
    assert sum(isinstance(f, LLMFunctionCallDetectedFrame) for f in down) == 1
