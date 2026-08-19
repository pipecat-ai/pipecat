#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI-compatible text handling around streamed tool calls.

Suppression is centralized in ``LLMService._push_llm_text``; these tests verify
the OpenAI streaming loop signals detection at the right time and that the
end-to-end frame flow matches the configured policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import ToolCallTextPolicy


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
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
    )


def _tool_call(*, index=0, call_id="call_1", name="lookup", arguments="{}"):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _service(chunks, *, policy=ToolCallTextPolicy.PRESERVE):
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(
                model="test-model",
                tool_call_text_policy=policy,
            )
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
async def test_text_is_preserved_by_default_before_and_after_a_tool_call():
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
async def test_text_after_a_tool_call_can_be_suppressed():
    service = _service(
        [
            _chunk(content="Before."),
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="After."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_any_tool_call_delta_starts_suppression():
    service = _service(
        [
            _chunk(
                tool_calls=[
                    _tool_call(call_id="call_1", name=None, arguments=None),
                ]
            ),
            _chunk(content="After an id-only tool delta."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == []


@pytest.mark.asyncio
async def test_parallel_tool_calls_are_unchanged_when_text_is_suppressed():
    service = _service(
        [
            _chunk(tool_calls=[_tool_call(call_id="call_1", name="first")]),
            _chunk(
                tool_calls=[
                    _tool_call(index=1, call_id="call_2", name="second"),
                ]
            ),
            _chunk(content="After."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == []
    function_calls = service.run_function_calls.await_args.args[0]
    assert [call.function_name for call in function_calls] == ["first", "second"]
    assert [call.tool_call_id for call in function_calls] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_suppression_state_resets_for_the_next_response():
    service = _service(
        [
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="Suppressed."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
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


@pytest.mark.asyncio
async def test_policy_can_be_updated_between_responses():
    chunks = [
        _chunk(tool_calls=[_tool_call()]),
        _chunk(content="After."),
    ]
    service = _service(chunks)

    pushed = await _run_and_collect(service, _context())
    assert pushed == ["After."]

    await service._update_settings(
        OpenAILLMService.Settings(tool_call_text_policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    )
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    pushed = await _run_and_collect(service, _context())
    assert pushed == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [
        ToolCallTextPolicy.PRESERVE,
        ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    ],
)
async def test_tool_call_takes_precedence_over_content_in_the_same_delta(policy):
    service = _service(
        [_chunk(content="Same delta.", tool_calls=[_tool_call()])],
        policy=policy,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == []
    service.run_function_calls.assert_awaited_once()
