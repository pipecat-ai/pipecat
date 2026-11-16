#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for function calls in AzureVoiceLiveLLMService.

Voice Live announces a call as a ``function_call`` item and sends its arguments
in ``response.function_call_arguments.done``; the service runs the call then.
Results come back through the context, either as regular tool messages or, for
tools registered with ``cancel_on_interruption=False``, as async-tool messages.
"""

import json
from typing import Any

import pytest

from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.azure.voice_live import events
from pipecat.services.azure.voice_live.llm import AzureVoiceLiveLLMService

CALL_ID = "call_1"


class _EventRecorder:
    """Records the client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def tool_outputs(self) -> list[tuple[str | None, str | None]]:
        return [
            (e.item.call_id, e.item.output)
            for e in self.events
            if isinstance(e, events.ConversationItemCreateEvent)
            and e.item.type == "function_call_output"
        ]

    def responses_created(self) -> int:
        return sum(isinstance(e, events.ResponseCreateEvent) for e in self.events)


class _ErrorRecorder:
    """Records the messages passed to ``push_error``."""

    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, error_msg: str, **kwargs):
        self.messages.append(error_msg)


class _ScriptedWebSocket:
    """Async-iterable websocket that yields scripted JSON strings."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


async def _ignore(*args, **kwargs) -> None:
    """Stand-in for methods that need a linked processor."""


def _make_service() -> tuple[AzureVoiceLiveLLMService, _EventRecorder, _ErrorRecorder]:
    """Construct a service with recorded client events and errors, and no connection."""
    service = AzureVoiceLiveLLMService(
        api_key="test-key",
        endpoint="https://my-resource.services.ai.azure.com",
    )
    sent = _EventRecorder()
    errors = _ErrorRecorder()
    service.send_client_event = sent
    service.push_error = errors
    service.push_frame = _ignore
    service.start_processing_metrics = _ignore
    service.start_ttfb_metrics = _ignore
    # The session is configured by the time tool results arrive.
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    return service, sent, errors


def _function_call_item(name: str = "get_current_weather") -> dict[str, Any]:
    return {
        "id": "item_call",
        "object": "realtime.item",
        "type": "function_call",
        "status": "in_progress",
        "call_id": CALL_ID,
        "name": name,
        "arguments": "",
    }


def _arguments_done(name: str | None, call_id: str = CALL_ID) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.done",
        "event_id": "event_args",
        "response_id": "resp_1",
        "item_id": "item_call",
        "output_index": 0,
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps({"location": "Hyderabad", "format": "celsius"}),
    }


async def _drive(service: AzureVoiceLiveLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server-event dicts through the receive handler."""
    service._websocket = _ScriptedWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


@pytest.mark.parametrize(
    "name_in_arguments_event",
    ["get_current_weather", None],
    ids=["named", "name-from-item"],
)
@pytest.mark.asyncio
async def test_a_function_call_runs_once_its_arguments_are_done(name_in_arguments_event):
    """The call item is announced twice, by conversation.item.created and
    response.output_item.added, and runs once."""
    service, _, _ = _make_service()
    service._context = LLMContext()
    calls = []

    async def _record_calls(function_calls):
        calls.extend(function_calls)

    service.run_function_calls = _record_calls

    await _drive(
        service,
        [
            {"type": "conversation.item.created", "event_id": "e1", "item": _function_call_item()},
            {
                "type": "response.output_item.added",
                "event_id": "e2",
                "response_id": "resp_1",
                "output_index": 0,
                "item": _function_call_item(),
            },
            _arguments_done(name_in_arguments_event),
        ],
    )

    assert len(calls) == 1
    assert calls[0].function_name == "get_current_weather"
    assert calls[0].tool_call_id == CALL_ID
    assert calls[0].arguments == {"location": "Hyderabad", "format": "celsius"}
    assert service._pending_function_calls == {}


@pytest.mark.asyncio
async def test_arguments_for_an_unannounced_call_run_nothing():
    service, _, _ = _make_service()
    service._context = LLMContext()
    calls = []

    async def _record_calls(function_calls):
        calls.extend(function_calls)

    service.run_function_calls = _record_calls

    await _drive(service, [_arguments_done("get_current_weather", call_id="call_unknown")])

    assert calls == []


@pytest.mark.asyncio
async def test_a_tool_result_is_sent_and_answered():
    service, sent, _ = _make_service()
    context = LLMContext()
    await service._handle_context(context)
    service._response_in_flight = False
    sent.events.clear()

    context.add_message({"role": "tool", "tool_call_id": CALL_ID, "content": '{"temp": 24}'})
    await service._handle_context(context)

    assert sent.tool_outputs() == [(CALL_ID, '{"temp": 24}')]
    assert sent.responses_created() == 1


@pytest.mark.asyncio
async def test_an_async_tool_final_result_is_sent_like_a_regular_one():
    """The started marker needs nothing sent: Voice Live is already waiting on the call."""
    service, sent, errors = _make_service()
    context = LLMContext([async_tool_messages.build_started_message(CALL_ID)])
    await service._handle_context(context)
    service._response_in_flight = False
    sent.events.clear()

    assert sent.tool_outputs() == []

    context.add_message(async_tool_messages.build_final_result_message(CALL_ID, '{"temp": 24}'))
    await service._handle_context(context)
    await service._handle_context(context)

    assert sent.tool_outputs() == [(CALL_ID, '{"temp": 24}')]
    assert sent.responses_created() == 1
    assert errors.messages == []


@pytest.mark.asyncio
async def test_an_async_tool_intermediate_result_is_dropped_with_an_error():
    """Voice Live has no channel for streamed results."""
    service, sent, errors = _make_service()
    context = LLMContext([async_tool_messages.build_started_message(CALL_ID)])
    await service._handle_context(context)
    service._response_in_flight = False
    sent.events.clear()

    context.add_message(
        async_tool_messages.build_intermediate_result_message(CALL_ID, '{"progress": 50}')
    )
    await service._handle_context(context)

    assert sent.tool_outputs() == []
    assert sent.responses_created() == 0
    assert len(errors.messages) == 1
    assert "streamed" in errors.messages[0]
