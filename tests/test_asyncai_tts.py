#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AsyncAITTSService runtime settings updates."""

import asyncio
import io
import json

import pytest
from loguru import logger

from pipecat.services.asyncai.tts import AsyncAITTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _service(**settings) -> AsyncAITTSService:
    return AsyncAITTSService(
        api_key="test-key",
        settings=AsyncAITTSService.Settings(model="m1", voice="v1", language=None, **settings),
    )


def _stub_reconnect(service: AsyncAITTSService) -> list[str]:
    """Record the reconnect sequence instead of touching the network."""
    calls: list[str] = []

    async def fake_disconnect():
        calls.append("disconnect")

    async def fake_connect():
        calls.append("connect")

    service._disconnect = fake_disconnect
    service._connect = fake_connect
    return calls


@pytest.mark.parametrize("field,value", [("voice", "v2"), ("model", "m2"), ("language", "es")])
def test_session_init_field_change_starts_a_new_session(field, value):
    # model, voice and language are only ever sent in the init message, so the
    # session has to be rebuilt for a change to them to reach Async at all.
    service = _service()
    calls = _stub_reconnect(service)

    asyncio.run(service._update_settings(AsyncAITTSService.Settings(**{field: value})))

    assert calls == ["disconnect", "connect"], f"{field} must rebuild the session"


def test_unchanged_settings_keep_the_session():
    service = _service()
    calls = _stub_reconnect(service)

    asyncio.run(service._update_settings(AsyncAITTSService.Settings(voice="v1")))

    assert calls == []


def test_new_session_carries_the_updated_voice():
    # The reconnect is only worth anything if the fresh init message actually
    # carries the new value, so drive the real _connect_websocket and read it.
    service = _service()
    sent: list[str] = []
    opened: list[int] = []

    class FakeWebsocket:
        state = None

        async def send(self, msg):
            sent.append(msg)

    async def fake_websocket_connect(_uri, **_kwargs):
        return FakeWebsocket()

    async def fake_call_event_handler(*_args, **_kwargs):
        pass

    service._websocket_connect = fake_websocket_connect
    service._call_event_handler = fake_call_event_handler

    async def run():
        await service.setup(frame_processor_setup(TaskManager()))
        await service._connect_websocket()
        assert json.loads(sent[-1])["voice"]["id"] == "v1", "sanity: first session uses v1"
        before = len(sent)
        # the real _disconnect/_connect run here, which is what re-sends init
        await service._update_settings(AsyncAITTSService.Settings(voice="v2"))
        opened.append(len(sent) - before)

    asyncio.run(run())

    assert opened == [1], "the settings change itself must open exactly one new session"
    assert json.loads(sent[-1])["voice"]["id"] == "v2", "the new session must carry the new voice"


def test_a_field_that_is_not_in_the_init_message_still_warns():
    # Anything outside model/voice/language genuinely cannot be applied to a
    # live session, so it must keep warning rather than look handled.
    service = _service()
    _stub_reconnect(service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        asyncio.run(service._update_settings(AsyncAITTSService.Settings(extra={"pace": 1.2})))
    finally:
        logger.remove(handler_id)

    assert "pace" in sink.getvalue()
