#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tool-response wire payloads across local Gemini Live reconnects (no API calls)."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


def result_messages(call_id, *, asynchronous=False):
    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "list_items", "arguments": "{}"},
            }
        ],
    }
    if asynchronous:
        return [
            call,
            async_tool_messages.build_started_message(call_id),
            async_tool_messages.build_final_result_message(call_id, '{"items": [7]}'),
        ]
    return [call, {"role": "tool", "tool_call_id": call_id, "content": '{"items": [7]}'}]


def sent(session):
    return [
        call.kwargs["function_responses"] for call in session.send_tool_response.await_args_list
    ]


async def push_context(service, context):
    await service.process_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)


@pytest.fixture
def service():
    llm = GeminiLiveLLMService(api_key="test")
    llm.register_function(
        "list_items", AsyncMock(), cancel_on_interruption=False, cancellable_by_llm=False
    )
    llm._session = AsyncMock()
    llm._context = LLMContext(messages=[{"role": "user", "content": "list them"}])
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_settled_results_not_replayed_on_resume(service, asynchronous):
    context = service._context
    first = service._session
    service._tool_call_id_to_name["call-1"] = "list_items"
    context.add_messages(result_messages("call-1", asynchronous=asynchronous))
    await push_context(service, context)
    assert [(r.id, r.name) for r in sent(first)] == [("call-1", "list_items")]

    service._session_resumption_handle = "test-resumption-handle"
    for call_id in ("call-2", "call-3"):
        session = AsyncMock()

        async def connect(session_resumption_handle=None, session=session):
            await service._handle_session_ready(session)

        service._connect = connect
        await service._reconnect()
        service._tool_call_id_to_name[call_id] = "list_items"
        context.add_messages(result_messages(call_id, asynchronous=asynchronous))
        await push_context(service, context)
        await push_context(service, context)
        assert [(r.id, r.name) for r in sent(session)] == [(call_id, "list_items")]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_before_ready", [False, True])
async def test_pending_async_result_keeps_name_and_scheduling(service, finish_before_ready):
    context = service._context
    service._tool_call_id_to_name["pending"] = "list_items"
    service._session_resumption_handle = "test-resumption-handle"
    session = AsyncMock()

    async def connect(session_resumption_handle=None):
        if finish_before_ready:
            context.add_messages(result_messages("pending", asynchronous=True))
        await service._handle_session_ready(session)

    service._connect = connect
    await service._reconnect()
    if not finish_before_ready:
        context.add_messages(result_messages("pending", asynchronous=True))
    await push_context(service, context)
    await push_context(service, context)
    responses = sent(session)
    assert len(responses) == 1
    response = responses[0]
    assert (response.id, response.name) == ("pending", "list_items")
    assert response.response == {"items": [7], "scheduling": "WHEN_IDLE"}
    assert response.scheduling.value == "WHEN_IDLE"
    assert response.will_continue is False


@pytest.mark.asyncio
async def test_fresh_reconnect_seeds_history(service):
    service._completed_tool_calls.add("old")
    service._tool_call_id_to_name["old"] = "list_items"
    service._context.add_messages(result_messages("old"))
    session = AsyncMock()

    async def connect(session_resumption_handle=None):
        assert session_resumption_handle is None
        await service._handle_session_ready(session)

    service._connect = connect
    await service._reconnect()
    assert not service._completed_tool_calls
    assert not service._tool_call_id_to_name
    assert session.send_client_content.await_count > 0
    assert not sent(session)


@pytest.mark.asyncio
async def test_ordinary_disconnect_clears_bookkeeping(service):
    service._session_resumption_handle = "test-resumption-handle"
    service._completed_tool_calls.add("old")
    service._tool_call_id_to_name["old"] = "list_items"
    await service._disconnect()
    assert not service._completed_tool_calls
    assert not service._tool_call_id_to_name
    assert service._session is None
