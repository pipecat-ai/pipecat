#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for stalled Responses WebSocket streams."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.tests.utils import run_test


def _make_service(**kwargs):
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(api_key="test-key", **kwargs)
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service._push_llm_text = AsyncMock()
    service._call_event_handler = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


def _socket(*events):
    """Deliver events with optional delays, then leave the connection open and silent."""
    pending = iter(events)

    async def recv():
        event = next(pending, None)
        while isinstance(event, (int, float)):
            await asyncio.sleep(event)
            event = next(pending, None)
        if event is None:
            await asyncio.Event().wait()
        return json.dumps(event)

    socket = AsyncMock()
    socket.recv.side_effect = recv
    return socket


def _context():
    return LLMContext(messages=[{"role": "user", "content": "hello"}])


def _completed():
    return {"type": "response.completed", "response": {"id": "resp_done", "model": "gpt-4.1"}}


def test_idle_timeout_defaults_and_opt_out():
    assert _make_service()._stream_idle_timeout_secs == 20.0
    assert _make_service(stream_idle_timeout_secs=None)._stream_idle_timeout_secs is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_idle_timeout(timeout):
    with pytest.raises(ValueError, match="stream_idle_timeout_secs"):
        _make_service(stream_idle_timeout_secs=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_on_timeout", [False, True])
async def test_partial_response_stall_ends_turn_without_replay(retry_on_timeout):
    service = _make_service(stream_idle_timeout_secs=0.05, retry_on_timeout=retry_on_timeout)
    socket = _socket(
        {"type": "response.created", "response": {"id": "resp_stalled"}},
        {"type": "response.output_text.delta", "delta": "Let me check"},
    )
    service._websocket = socket
    service._store_previous_response_state("resp_previous", [], [])

    await asyncio.wait_for(
        service.process_frame(LLMContextFrame(_context()), FrameDirection.DOWNSTREAM), timeout=1
    )

    service._push_llm_text.assert_awaited_once_with("Let me check")
    service._call_event_handler.assert_any_await("on_completion_timeout")
    service.push_error.assert_awaited_once()
    assert isinstance(service.push_error.await_args.kwargs["exception"], TimeoutError)
    assert [type(call.args[0]) for call in service.push_frame.await_args_list] == [
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    ]
    assert socket.send.await_count == 1
    socket.close.assert_awaited_once()
    assert service._websocket is None
    assert service._previous_response_id is None
    assert service._current_response_id is None
    assert not service._needs_drain
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "response.created", "response": {"id": "resp_stalled"}}],
        [{"type": "response.in_progress"}],
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f"},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"x":'},
        ],
    ],
    ids=["before-first-event", "after-created", "after-acknowledgement", "partial-tool-call"],
)
async def test_stall_without_text_is_bounded(events):
    service = _make_service(stream_idle_timeout_secs=0.05)
    service._websocket = socket = _socket(*events)

    await asyncio.wait_for(
        service.process_frame(LLMContextFrame(_context()), FrameDirection.DOWNSTREAM), timeout=1
    )

    service._call_event_handler.assert_any_await("on_completion_timeout")
    service.push_error.assert_awaited_once()
    service.run_function_calls.assert_not_awaited()
    socket.close.assert_awaited_once()
    assert socket.send.await_count == 1


@pytest.mark.asyncio
async def test_stalled_reasoning_closes_thought_frames():
    service = _make_service(stream_idle_timeout_secs=0.05)
    service._websocket = _socket(
        {"type": "response.reasoning_summary_text.delta", "delta": "Thinking"}
    )

    await asyncio.wait_for(
        service.process_frame(LLMContextFrame(_context()), FrameDirection.DOWNSTREAM), timeout=1
    )

    assert [type(call.args[0]) for call in service.push_frame.await_args_list] == [
        LLMFullResponseStartFrame,
        LLMThoughtStartFrame,
        LLMThoughtTextFrame,
        LLMThoughtEndFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_healthy_stream_can_outlast_idle_timeout():
    service = _make_service(stream_idle_timeout_secs=0.2)
    events = []
    for i in range(6):
        events.extend([0.05, {"type": "response.output_text.delta", "delta": str(i)}])
    service._websocket = socket = _socket(*events, _completed())

    await asyncio.wait_for(service._process_context(_context()), timeout=2)

    assert [call.args[0] for call in service._push_llm_text.await_args_list] == list("012345")
    socket.close.assert_not_awaited()
    service.push_error.assert_not_awaited()
    assert service._previous_response_id == "resp_done"


@pytest.mark.asyncio
async def test_idle_timeout_can_be_disabled():
    service = _make_service(
        stream_idle_timeout_secs=None, retry_on_timeout=True, retry_timeout_secs=0.01
    )
    service._websocket = socket = _socket(
        {"type": "response.output_text.delta", "delta": "hello"}, 0.1, _completed()
    )

    await asyncio.wait_for(service._process_context(_context()), timeout=1)

    socket.close.assert_not_awaited()
    service._push_llm_text.assert_awaited_once_with("hello")
    service.push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_output_retry_still_runs_once_and_retry_is_bounded():
    service = _make_service(
        stream_idle_timeout_secs=0.05, retry_on_timeout=True, retry_timeout_secs=0.03
    )
    first = _socket()
    second = _socket()
    service._websocket = first
    service._websocket_connect = AsyncMock(return_value=second)

    await asyncio.wait_for(
        service.process_frame(LLMContextFrame(_context()), FrameDirection.DOWNSTREAM), timeout=1
    )

    assert first.send.await_count == second.send.await_count == 1
    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
    service._websocket_connect.assert_awaited_once()
    service._call_event_handler.assert_any_await("on_completion_timeout")
    service.push_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_output_deadline_takes_precedence_over_idle_timeout():
    service = _make_service(
        stream_idle_timeout_secs=0.03, retry_on_timeout=True, retry_timeout_secs=0.3
    )
    service._websocket = socket = _socket(
        0.06, {"type": "response.output_text.delta", "delta": "hello"}, _completed()
    )

    await asyncio.wait_for(service._process_context(_context()), timeout=1)

    assert socket.send.await_count == 1
    socket.close.assert_not_awaited()
    service._push_llm_text.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_interruption_is_not_reported_as_an_idle_timeout():
    service = _make_service(stream_idle_timeout_secs=0.05)
    service._websocket = socket = _socket()
    socket.recv.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.process_frame(LLMContextFrame(_context()), FrameDirection.DOWNSTREAM)

    service._call_event_handler.assert_not_awaited()
    service.push_error.assert_not_awaited()
    socket.close.assert_not_awaited()
    assert service._needs_drain


@pytest.mark.asyncio
async def test_pipeline_recovers_on_a_new_socket_after_a_stalled_turn():
    """Exercise real socket I/O and queued context frames without a cloud provider."""
    requests = []
    connections = []

    async def handler(socket):
        connections.append(socket)
        requests.append(json.loads(await socket.recv()))
        if len(requests) == 1:
            await socket.send(
                json.dumps({"type": "response.output_text.delta", "delta": "partial"})
            )
            await socket.wait_closed()
        else:
            await socket.send(json.dumps({"type": "response.output_text.delta", "delta": "fresh"}))
            await socket.send(json.dumps(_completed()))
            await socket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        with patch.object(OpenAIResponsesLLMService, "_create_client"):
            service = OpenAIResponsesLLMService(
                api_key="test-key",
                ws_url=f"ws://127.0.0.1:{port}",
                stream_idle_timeout_secs=0.1,
                retry_on_timeout=True,
            )
        timeouts = []

        @service.event_handler("on_completion_timeout")
        async def on_timeout(service):
            timeouts.append(True)

        down, up = await asyncio.wait_for(
            run_test(
                service,
                frames_to_send=[
                    LLMContextFrame(_context()),
                    LLMContextFrame(LLMContext([{"role": "user", "content": "next turn"}])),
                ],
            ),
            timeout=5,
        )

    assert [type(frame) for frame in down] == [
        LLMServiceMetadataFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert [frame.text for frame in down if isinstance(frame, LLMTextFrame)] == ["partial", "fresh"]
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert isinstance(errors[0].exception, TimeoutError)
    assert timeouts == [True]
    assert len(connections) == len(requests) == 2
    assert connections[0] is not connections[1]
    assert requests[1]["input"] == [{"role": "user", "content": "next turn"}]
    assert "previous_response_id" not in requests[1]
