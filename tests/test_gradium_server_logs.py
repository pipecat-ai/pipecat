#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium server log messages: asked for with ``log_level``, logged at their level."""

import json
from unittest.mock import AsyncMock

import pytest
from loguru import logger
from websockets.protocol import State

from pipecat.services.gradium.stt import GradiumSTTService
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.gradium.utils import log_server_message

LOG = {"type": "log", "level": "warning", "message": "unused json_config keys: tmep"}


@pytest.fixture
def logged():
    """The (level, message) pairs logged during the test."""
    records: list[tuple[str, str]] = []
    sink = logger.add(lambda m: records.append((m.record["level"].name, m.record["message"])))
    yield records
    logger.remove(sink)


class _Socket:
    """A websocket that plays back `messages` to both recv() and iteration."""

    state = State.OPEN

    def __init__(self, messages: list[dict]):
        self._messages = [json.dumps(m) for m in messages]
        self.send = AsyncMock()

    async def recv(self) -> str:
        return self._messages.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("unheard-of", "WARNING"),
    ],
)
def test_a_server_log_is_logged_at_its_own_level(logged, level, expected):
    log_server_message({"type": "log", "level": level, "message": "m"})

    assert logged == [(expected, f"Gradium {expected.lower()}: m")]


@pytest.mark.parametrize(("log_level", "expected"), [(None, None), ("warning", "warning")])
@pytest.mark.asyncio
async def test_stt_asks_for_server_logs_only_when_enabled(log_level, expected, logged):
    service = GradiumSTTService(api_key="test-key", sample_rate=16000, log_level=log_level)
    socket = _Socket([LOG, {"type": "ready"}])
    service._websocket_connect = AsyncMock(return_value=socket)
    service._call_event_handler = AsyncMock()

    await service._connect_websocket()

    setup = json.loads(socket.send.call_args_list[0].args[0])
    assert setup.get("log_level") == expected
    # a log that comes before ``ready`` is logged, and the connection still completes
    assert service._websocket is socket
    assert ("WARNING", "Gradium warning: unused json_config keys: tmep") in logged


@pytest.mark.asyncio
async def test_stt_logs_a_server_log_received_during_the_session(logged):
    service = GradiumSTTService(api_key="test-key", sample_rate=16000, log_level="warning")
    service.push_error = AsyncMock()
    service._websocket = _Socket([LOG, {"type": "end_of_stream"}])

    await service._receive_messages()

    assert ("WARNING", "Gradium warning: unused json_config keys: tmep") in logged
    service.push_error.assert_not_called()


@pytest.mark.parametrize(("log_level", "expected"), [(None, None), ("error", "error")])
def test_tts_asks_for_server_logs_only_when_enabled(log_level, expected):
    service = GradiumTTSService(api_key="test-key", log_level=log_level)

    assert service._build_setup_msg("ctx").get("log_level") == expected


@pytest.mark.asyncio
async def test_tts_logs_a_server_log_received_during_the_session(logged):
    service = GradiumTTSService(api_key="test-key", log_level="warning")
    service.push_error = AsyncMock()
    service.push_frame = AsyncMock()
    service._websocket = _Socket([{**LOG, "client_req_id": "ctx"}, {"type": "ready"}])

    await service._receive_messages()

    assert ("WARNING", "Gradium warning: unused json_config keys: tmep") in logged
    service.push_error.assert_not_called()
    service.push_frame.assert_not_called()
