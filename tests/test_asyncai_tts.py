#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AsyncAITTSService runtime settings updates."""

import asyncio
import io
import json
from unittest.mock import AsyncMock

import pytest
from loguru import logger
from websockets.protocol import State

from pipecat.services.asyncai.tts import AsyncAITTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _service() -> AsyncAITTSService:
    return AsyncAITTSService(
        api_key="test-key",
        settings=AsyncAITTSService.Settings(model="m1", voice="v1", language=None),
    )


def _stub_reconnect(monkeypatch, service: AsyncAITTSService) -> list[str]:
    """Record the reconnect sequence in place of the real websocket calls."""
    calls: list[str] = []
    monkeypatch.setattr(
        service, "_disconnect", AsyncMock(side_effect=lambda: calls.append("disconnect"))
    )
    monkeypatch.setattr(service, "_connect", AsyncMock(side_effect=lambda: calls.append("connect")))
    return calls


class FakeWebsocket:
    """Websocket that records what the service sends and never yields a message.

    The receive loop reconnects on its own whenever it sees the connection
    drop, so the iterator parks until close instead of ending: a socket that
    stops iterating would leave the base class opening sessions of its own,
    on top of the ones under test.
    """

    state = State.OPEN

    def __init__(self, sent: list[str]):
        self._sent = sent
        self._closed = asyncio.Event()

    async def send(self, msg: str):
        self._sent.append(msg)

    async def ping(self):
        pass

    async def close(self):
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("voice", "v2"), ("model", "m2"), ("language", "es")])
async def test_session_init_field_change_starts_a_new_session(monkeypatch, field, value):
    # model, voice and language are only ever sent in the init message, so the
    # session has to be rebuilt for a change to them to reach Async at all.
    service = _service()
    calls = _stub_reconnect(monkeypatch, service)

    await service._update_settings(AsyncAITTSService.Settings(**{field: value}))

    assert calls == ["disconnect", "connect"], f"{field} must rebuild the session"


@pytest.mark.asyncio
async def test_unchanged_settings_keep_the_session(monkeypatch):
    service = _service()
    calls = _stub_reconnect(monkeypatch, service)

    await service._update_settings(AsyncAITTSService.Settings(voice="v1"))

    assert calls == []


@pytest.mark.asyncio
async def test_new_session_carries_the_updated_voice(monkeypatch):
    # The reconnect is only worth anything if the fresh init message actually
    # carries the new value, so drive the real _connect_websocket and read it.
    service = _service()
    sent: list[str] = []

    async def fake_websocket_connect(_uri, **_kwargs):
        return FakeWebsocket(sent)

    monkeypatch.setattr(service, "_websocket_connect", fake_websocket_connect)

    await service.setup(frame_processor_setup(TaskManager()))
    try:
        assert json.loads(sent[-1])["voice"]["id"] == "v1"
        before = len(sent)

        await service._update_settings(AsyncAITTSService.Settings(voice="v2"))

        assert len(sent) - before == 1, "the settings change must open exactly one new session"
        assert json.loads(sent[-1])["voice"]["id"] == "v2"
    finally:
        await service.cleanup()


@pytest.mark.asyncio
async def test_a_field_that_is_not_in_the_init_message_still_warns(monkeypatch):
    # Anything outside model/voice/language genuinely cannot be applied to a
    # live session, so it must keep warning rather than look handled.
    service = _service()
    _stub_reconnect(monkeypatch, service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._update_settings(AsyncAITTSService.Settings(extra={"pace": 1.2}))
    finally:
        logger.remove(handler_id)

    assert "pace" in sink.getvalue()
