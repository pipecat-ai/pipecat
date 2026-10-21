#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how InworldTTSService ends its server-side contexts."""

import asyncio
import json

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.tests.utils import SleepFrame, run_test


class FakeWebsocket:
    """Open socket that never receives anything until it is closed."""

    def __init__(self):
        self.state = State.OPEN
        self._closed = asyncio.Event()

    async def close(self):
        self.state = State.CLOSED
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration


def _make_service():
    service = InworldTTSService(api_key="test-key")
    sent = []

    async def record(message, _report_error):
        sent.append(json.loads(message))

    async def connect():
        service._websocket = FakeWebsocket()

    service.send_with_retry = record
    service._connect_websocket = connect
    service._websocket = FakeWebsocket()
    return service, sent


async def _drain(gen):
    return [frame async for frame in gen]


def _kinds(sent):
    return [next(k for k in m if k != "contextId") for m in sent]


@pytest.mark.asyncio
async def test_completed_turn_closes_context():
    service, sent = _make_service()

    service._turn_context_id = "ctx-1"
    await _drain(service.run_tts("Hello there. ", "ctx-1"))
    await service.on_turn_context_completed()

    assert _kinds(sent)[-1] == "close_context"
    assert "cancel_context" not in _kinds(sent)


@pytest.mark.asyncio
async def test_interruption_cancels_context():
    service, sent = _make_service()

    await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello there. "),
            LLMTextFrame("How are"),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
        ],
    )

    assert _kinds(sent) == ["create", "send_text", "cancel_context"]
    assert len({m["contextId"] for m in sent}) == 1


@pytest.mark.asyncio
async def test_interruption_after_turn_completed_cancels_context():
    service, sent = _make_service()

    await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello there. How are you? "),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
        ],
    )

    assert _kinds(sent)[-2:] == ["close_context", "cancel_context"]
    assert len({m["contextId"] for m in sent}) == 1


@pytest.mark.asyncio
async def test_disconnect_cancels_open_contexts():
    service, sent = _make_service()

    await _drain(service.run_tts("Hello there. ", "ctx-1"))
    await service._disconnect_websocket()

    assert sent[-1] == {"cancel_context": {}, "contextId": "ctx-1"}
