#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the WebsocketTTSService keepalive."""

import asyncio
from unittest.mock import MagicMock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

from pipecat.services.tts_service import WebsocketTTSService
from pipecat.utils.asyncio.task_manager import TaskManager

KEEPALIVE_INTERVAL = 0.01


class KeepaliveTTSService(WebsocketTTSService):
    """Minimal websocket TTS service that records its keepalives."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keepalives = 0
        self.failures_left = 0

    async def run_tts(self, text: str, context_id: str):
        yield None

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass

    async def _send_keepalive(self):
        if self.failures_left:
            self.failures_left -= 1
            raise ConnectionClosedError(None, None)
        self.keepalives += 1


def _service(keepalive_interval: float | None = KEEPALIVE_INTERVAL) -> KeepaliveTTSService:
    service = KeepaliveTTSService(keepalive_interval=keepalive_interval)
    service._task_manager = TaskManager()
    service._websocket = MagicMock(state=State.OPEN)
    return service


async def _wait_for_keepalives(service: KeepaliveTTSService, count: int):
    async with asyncio.timeout(1):
        while service.keepalives < count:
            await asyncio.sleep(KEEPALIVE_INTERVAL)


@pytest.mark.asyncio
async def test_keepalive_disabled_by_default():
    """Without a keepalive interval, no keepalive task is started."""
    service = _service(keepalive_interval=None)
    service._create_keepalive_task()
    assert service._keepalive_task is None


@pytest.mark.asyncio
async def test_keepalive_sent_periodically():
    """Keepalives are sent every interval while the websocket is open."""
    service = _service()
    service._create_keepalive_task()
    try:
        await _wait_for_keepalives(service, 3)
    finally:
        await service._cancel_keepalive_task()
    assert service._keepalive_task is None


@pytest.mark.asyncio
async def test_keepalive_survives_failed_send():
    """A failed send is logged and later keepalives still go out."""
    service = _service()
    service.failures_left = 2
    service._create_keepalive_task()
    try:
        await _wait_for_keepalives(service, 2)
        assert service.failures_left == 0
        assert not service._keepalive_task.done()
    finally:
        await service._cancel_keepalive_task()


@pytest.mark.asyncio
async def test_keepalive_skipped_while_websocket_not_open():
    """No keepalive is sent while the websocket is closed, and sending resumes once it reopens."""
    service = _service()
    service._websocket.state = State.CLOSED
    service._create_keepalive_task()
    try:
        await asyncio.sleep(KEEPALIVE_INTERVAL * 5)
        assert service.keepalives == 0
        service._websocket.state = State.OPEN
        await _wait_for_keepalives(service, 1)
    finally:
        await service._cancel_keepalive_task()


@pytest.mark.asyncio
async def test_create_keepalive_task_keeps_running_task():
    """Creating the task while one is running leaves the running task in place."""
    service = _service()
    service._create_keepalive_task()
    task = service._keepalive_task
    try:
        service._create_keepalive_task()
        assert service._keepalive_task is task
    finally:
        await service._cancel_keepalive_task()


@pytest.mark.asyncio
async def test_create_keepalive_task_replaces_finished_task():
    """A finished keepalive task is replaced by a new one."""
    service = _service()

    async def finished():
        pass

    service._keepalive_task = service.create_task(finished(), name="finished")
    await service._keepalive_task
    service._create_keepalive_task()
    try:
        assert not service._keepalive_task.done()
        await _wait_for_keepalives(service, 1)
    finally:
        await service._cancel_keepalive_task()
