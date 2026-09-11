#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for node-transition function-call deferral in OpenAI LLM services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService


def _make_service() -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        return OpenAILLMService(api_key="test-key")


def _make_function_call(name: str, tool_call_id: str) -> FunctionCallFromLLM:
    return FunctionCallFromLLM(
        context=LLMContext(),
        tool_call_id=tool_call_id,
        function_name=name,
        arguments={},
    )


@pytest.mark.asyncio
async def test_non_transition_call_is_not_deferred_after_generated_text():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call("look_up_account", "call-ordinary")

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_node_transition_call_is_deferred_after_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == [function_call]


@pytest.mark.asyncio
async def test_node_transition_call_runs_immediately_without_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=False,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_mixed_batch_with_node_transition_is_deferred_together():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_calls = [
        _make_function_call("look_up_account", "call-ordinary"),
        _make_function_call("transition_to_next_node", "call-transition"),
    ]

    await service._run_or_defer_function_calls(
        function_calls,
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == function_calls


@pytest.mark.asyncio
async def test_pending_node_transition_batch_runs_after_tts():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )
    service._pending_node_transition_function_calls = [function_call]

    await service._run_pending_node_transition_function_calls()

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


# --- Streaming-level tests for the deferral decision -------------------------
#
# The deferral is only released by a BotStoppedSpeakingFrame, so a completion
# whose content never reaches the synthesizer (whitespace, bare punctuation,
# markup) must not arm it — models that emit such fragments alongside a
# node-transition tool call would otherwise park the transition forever and
# leave the call silent.


class _FakeStream:
    """Stands in for the provider's chat completion stream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _content_chunk(text: str):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=None, content=text))],
    )


def _tool_call_chunk(name: str, tool_call_id: str = "call-transition"):
    tool_call = SimpleNamespace(
        index=0,
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments="{}"),
    )
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[tool_call], content=None))],
    )


def _streaming_service(chunks) -> OpenAILLMService:
    service = _make_service()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_ttfat_metrics = AsyncMock()
    service._push_llm_text = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    return service


@pytest.mark.asyncio
async def test_whitespace_only_content_does_not_defer_node_transition():
    service = _streaming_service(
        [_content_chunk("\n"), _tool_call_chunk("transition_to_next_node")]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_awaited_once()
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_punctuation_only_content_does_not_defer_node_transition():
    service = _streaming_service(
        [_content_chunk("..."), _tool_call_chunk("transition_to_next_node")]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_awaited_once()
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_markup_only_content_does_not_defer_node_transition():
    service = _streaming_service(
        [_content_chunk("<break/>"), _tool_call_chunk("transition_to_next_node")]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_awaited_once()
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_speakable_content_defers_node_transition():
    service = _streaming_service(
        [_content_chunk("ठीक है।"), _tool_call_chunk("transition_to_next_node")]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_not_awaited()
    pending = service._pending_node_transition_function_calls
    assert len(pending) == 1
    assert pending[0].function_name == "transition_to_next_node"


@pytest.mark.asyncio
async def test_speakable_content_split_across_chunks_defers_node_transition():
    service = _streaming_service(
        [
            _content_chunk(" "),
            _content_chunk("ok"),
            _tool_call_chunk("transition_to_next_node"),
        ]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_not_awaited()
    assert len(service._pending_node_transition_function_calls) == 1
