#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the WebSocket client transport."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import pipecat.transports.websocket.client as websocket_client
from pipecat.transports.websocket.client import (
    WebsocketClientCallbacks,
    WebsocketClientParams,
    WebsocketClientSession,
)
from pipecat.utils.asyncio.task_manager import TaskManager


class _FakeWebsocket:
    """A connection that carries no messages and stays open until closed."""

    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


def _make_session(monkeypatch) -> tuple[WebsocketClientSession, list[_FakeWebsocket], AsyncMock]:
    opened = []

    async def fake_connect(**kwargs):
        await asyncio.sleep(0.01)  # the real one dials the server
        websocket = _FakeWebsocket()
        opened.append(websocket)
        return websocket

    monkeypatch.setattr(websocket_client, "websocket_connect", fake_connect)

    on_connected = AsyncMock()
    callbacks = WebsocketClientCallbacks(
        on_connected=on_connected,
        on_disconnected=AsyncMock(),
        on_message=AsyncMock(),
    )
    session = WebsocketClientSession("ws://example.com", WebsocketClientParams(), callbacks, "Test")
    return session, opened, on_connected


@pytest.mark.asyncio
async def test_concurrent_setup_opens_a_single_websocket(monkeypatch):
    """The input and output transports share one session, and both connect it.

    They are set up concurrently, so a socket opened per caller would leave the
    losing one dialled with nobody reading it, its handler task overwritten.
    """
    session, opened, on_connected = _make_session(monkeypatch)

    task_manager = TaskManager()
    await session.setup(task_manager)
    await session.setup(task_manager)
    await asyncio.gather(session.connect(), session.connect())

    assert len(opened) == 1, f"{len(opened)} websockets opened, so one goes unread"
    on_connected.assert_awaited_once()

    await session.disconnect()
    await session.disconnect()


@pytest.mark.asyncio
async def test_the_websocket_outlives_the_first_transport_to_disconnect(monkeypatch):
    """Closing on the first disconnect would leave the other transport sending
    over a closed socket."""
    session, opened, _ = _make_session(monkeypatch)

    task_manager = TaskManager()
    await session.setup(task_manager)
    await session.setup(task_manager)
    await asyncio.gather(session.connect(), session.connect())

    await session.disconnect()
    assert not opened[0].closed

    await session.disconnect()
    assert opened[0].closed
