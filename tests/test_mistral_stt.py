#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how MistralSTTService handles a connection the server has closed.

The SDK marks a dropped socket closed before the event iterator it is feeding
ends, so the service can hold a connection that is already closed. Both the
send path and the teardown path have to recognize that.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("mistralai")

from pipecat.services.mistral.stt import MistralSTTService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class FakeConnection:
    """Stands in for the SDK's RealtimeConnection."""

    def __init__(self):
        self.is_closed = False
        self.sent: list[bytes] = []
        self.close_calls = 0
        self._release = asyncio.Event()

    async def events(self):
        # Ends only once the test releases it, so `is_closed` can be observed
        # while the service still holds the connection.
        await self._release.wait()
        return
        yield  # pragma: no cover

    def release(self):
        self._release.set()

    async def send_audio(self, audio: bytes):
        if self.is_closed:
            raise RuntimeError("cannot send on a closed connection")
        self.sent.append(audio)

    async def close(self):
        self.close_calls += 1
        self.is_closed = True

    async def flush_audio(self):
        pass


async def _service(connection: FakeConnection):
    """A started service whose SDK client hands out `connection`."""
    service = MistralSTTService(api_key="test-key")
    state = {"connect_fails": False}

    async def connect(**kwargs):
        if state["connect_fails"]:
            raise RuntimeError("server unreachable")
        return connection

    service._client = SimpleNamespace(
        audio=SimpleNamespace(realtime=SimpleNamespace(connect=connect))
    )
    await service.setup(frame_processor_setup(TaskManager(), audio_in_sample_rate=16000))
    return service, state


@pytest.mark.asyncio
async def test_audio_is_dropped_when_the_reconnect_fails():
    """A chunk is dropped rather than sent on a connection that is closed."""
    connection = FakeConnection()
    service, state = await _service(connection)

    connection.is_closed = True
    state["connect_fails"] = True

    async for _ in service.run_stt(b"\x00\x01" * 160):
        pass

    assert connection.sent == []
    connection.release()


@pytest.mark.asyncio
async def test_teardown_releases_a_connection_that_is_already_closed():
    """Teardown announces the disconnect and lets go of the closed connection."""
    connection = FakeConnection()
    service, _ = await _service(connection)

    disconnected = asyncio.Event()

    @service.event_handler("on_disconnected")
    async def _on_disconnected(_service):
        disconnected.set()

    connection.is_closed = True
    await service.cleanup()

    assert service._connection is None
    # Event handlers run in their own task, so wait for it rather than racing it.
    await asyncio.wait_for(disconnected.wait(), timeout=1)
    # Nothing to close: the server already did.
    assert connection.close_calls == 0
    connection.release()


@pytest.mark.asyncio
async def test_teardown_closes_a_connection_that_is_still_open():
    """An open connection is closed exactly once on teardown."""
    connection = FakeConnection()
    service, _ = await _service(connection)

    await service.cleanup()

    assert service._connection is None
    assert connection.close_calls == 1
    connection.release()
