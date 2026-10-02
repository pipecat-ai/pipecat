#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how OpenAIRealtimeLLMService reports connection problems.

An error event leaves the session open, so the service reports it and keeps
reading. A socket that closes, a send that fails on a live session and a
connection that cannot be opened leave the service with nothing to work with,
so each is reported as permanent. ``AzureRealtimeLLMService`` shares the
receive and send paths and has its own ``_connect``.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from pipecat.services.azure.realtime import llm as azure_realtime_llm
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.openai.realtime import llm as realtime_llm
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

ERROR_EVENT = {
    "type": "error",
    "event_id": "event_1",
    "error": {
        "type": "invalid_request_error",
        "code": "invalid_value",
        "message": "Invalid value.",
    },
}
SPEECH_STARTED_EVENT = {
    "type": "input_audio_buffer.speech_started",
    "event_id": "event_2",
    "audio_start_ms": 1000,
    "item_id": "item_1",
}


def _closed() -> ConnectionClosedError:
    return ConnectionClosedError(Close(1011, "internal error"), None)


class _FakeWebSocket:
    """Yields scripted server events, then ends or raises ``drop``."""

    def __init__(self, events=(), drop: Exception | None = None):
        self._messages = [json.dumps(event) for event in events]
        self._drop = drop

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        if self._drop:
            raise self._drop
        raise StopAsyncIteration

    async def send(self, data):
        if self._drop:
            raise self._drop


def _make_service() -> OpenAIRealtimeLLMService:
    service = OpenAIRealtimeLLMService(api_key="test-key")
    service.push_error = AsyncMock()
    return service


def _is_permanent(push_error: AsyncMock) -> bool:
    return push_error.await_args.kwargs.get("force_treat_as_permanent") is True


@pytest.mark.asyncio
async def test_an_error_event_is_reported_and_the_next_event_is_still_read():
    service = _make_service()
    service._handle_evt_speech_started = AsyncMock()
    service._websocket = _FakeWebSocket([ERROR_EVENT, SPEECH_STARTED_EVENT])

    await service._receive_task_handler()

    service.push_error.assert_awaited_once()
    assert not _is_permanent(service.push_error)
    service._handle_evt_speech_started.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_socket_closed_by_the_server_is_reported_as_permanent():
    service = _make_service()
    service._websocket = _FakeWebSocket(drop=_closed())

    await service._receive_task_handler()

    service.push_error.assert_awaited_once()
    assert _is_permanent(service.push_error)


@pytest.mark.asyncio
async def test_a_socket_closed_while_disconnecting_is_not_an_error():
    service = _make_service()
    service._websocket = _FakeWebSocket(drop=_closed())
    service._disconnecting = True

    await service._receive_task_handler()

    service.push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_send_on_a_live_session_is_reported_as_permanent():
    service = _make_service()
    service._websocket = _FakeWebSocket(drop=_closed())

    await service._ws_send({"type": "input_audio_buffer.clear"})

    service.push_error.assert_awaited_once()
    assert _is_permanent(service.push_error)


@pytest.mark.asyncio
async def test_a_connection_that_cannot_be_opened_is_reported_as_permanent():
    service = _make_service()
    connect = AsyncMock(side_effect=ConnectionRefusedError(111, "Connection refused"))

    with patch.object(realtime_llm, "websocket_connect", connect):
        await service._connect()

    service.push_error.assert_awaited_once()
    assert _is_permanent(service.push_error)
    assert service._websocket is None


@pytest.mark.asyncio
async def test_an_azure_connection_that_cannot_be_opened_is_reported_as_permanent():
    service = AzureRealtimeLLMService(
        base_url="wss://example.openai.azure.com/openai/v1/realtime", api_key="test-key"
    )
    service.push_error = AsyncMock()
    connect = AsyncMock(side_effect=ConnectionRefusedError(111, "Connection refused"))

    with patch.object(azure_realtime_llm, "websocket_connect", connect):
        await service._connect()

    service.push_error.assert_awaited_once()
    assert _is_permanent(service.push_error)
    assert service._websocket is None
