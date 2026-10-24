#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text handling around streamed SambaNova tool calls.

SambaNova has its own streaming loop with tool-call argument handling that
differs from the OpenAI base (no tool-call indexing). Suppression is centered
in ``LLMService._push_llm_text``; these tests verify the loop signals
detection at the right time and that the frame flow matches the policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.sambanova.llm import SambaNovaLLMService
from pipecat.services.settings import ToolCallPolicy


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

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
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
    )


def _tool_call(*, call_id="call_1", name="lookup", arguments="{}"):
    # SambaNova tool-call deltas carry no index.
    return SimpleNamespace(
        index=0,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _service(chunks, *, policy=ToolCallPolicy.PRESERVE_TEXT_AFTER_TOOL_CALL_DETECTED):
    with patch.object(SambaNovaLLMService, "create_client"):
        service = SambaNovaLLMService(
            api_key="test-key",
            settings=SambaNovaLLMService.Settings(
                model="test-model",
                tool_call_text_policy=policy,
            ),
        )
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


def _context():
    return LLMContext(messages=[{"role": "user", "content": "Look it up"}])


async def _run_and_collect(service, context):
    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(context)
    return pushed


@pytest.mark.asyncio
async def test_sambanova_text_is_preserved_by_default_before_and_after_a_tool_call():
    service = _service(
        [
            _chunk(content="Before."),
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="After."),
        ]
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before.", "After."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_sambanova_text_after_a_tool_call_can_be_suppressed():
    service = _service(
        [
            _chunk(content="Before."),
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="After."),
        ],
        policy=ToolCallPolicy.SUPPRESS_TEXT_AFTER_TOOL_CALL_DETECTED,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_sambanova_suppression_state_resets_for_the_next_response():
    service = _service(
        [
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="Suppressed."),
        ],
        policy=ToolCallPolicy.SUPPRESS_TEXT_AFTER_TOOL_CALL_DETECTED,
    )

    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(_context())
        assert pushed == []

        # The next response's start frame resets suppression state.
        await service.push_frame(LLMFullResponseStartFrame())
        service.get_chat_completions = AsyncMock(
            return_value=_FakeStream([_chunk(content="Final answer.")])
        )
        await service._process_context(_context())

    assert pushed == ["Final answer."]
