#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for InworldTTSService context lifecycle."""

import json

import pytest
from websockets.protocol import State

from pipecat.services.inworld.tts import InworldTTSService


class FakeWebsocket:
    state = State.OPEN


def _make_service():
    service = InworldTTSService(api_key="test-key")
    sent = []

    async def record(message, _report_error):
        sent.append(json.loads(message))

    service.send_with_retry = record
    service._websocket = FakeWebsocket()
    return service, sent


async def _drain(gen):
    return [frame async for frame in gen]


@pytest.mark.asyncio
async def test_context_is_created_with_first_text():
    service, sent = _make_service()

    await service.on_turn_context_created("ctx-1")
    assert sent == []

    await _drain(service.run_tts("Hello there. ", "ctx-1"))
    await _drain(service.run_tts("How are you? ", "ctx-1"))

    assert [list(m)[0] for m in sent] == ["create", "send_text", "send_text"]
    assert all(m["contextId"] == "ctx-1" for m in sent)


@pytest.mark.asyncio
async def test_turn_without_text_sends_nothing():
    service, sent = _make_service()

    service._turn_context_id = service.create_context_id()
    await service.on_turn_context_created(service._turn_context_id)
    await service.on_turn_context_completed()

    assert sent == []


@pytest.mark.asyncio
async def test_turn_with_text_is_flushed_and_closed():
    service, sent = _make_service()

    service._turn_context_id = "ctx-1"
    await _drain(service.run_tts("Hello there. ", "ctx-1"))
    await service.on_turn_context_completed()

    assert [list(m)[0] for m in sent] == ["create", "send_text", "flush_context", "close_context"]
