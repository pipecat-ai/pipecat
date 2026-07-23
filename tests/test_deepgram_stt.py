#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from pipecat.services.deepgram.stt import DeepgramSTTService, _derive_deepgram_urls


@pytest.mark.parametrize(
    "base_url, expected_ws, expected_http",
    [
        # Secure schemes
        ("wss://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        ("https://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        # Insecure schemes (air-gapped deployments)
        ("ws://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"),
        ("http://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"),
        # Bare hostname defaults to secure
        ("mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        # With port
        ("ws://localhost:8080", "ws://localhost:8080", "http://localhost:8080"),
        ("wss://localhost:443", "wss://localhost:443", "https://localhost:443"),
        ("localhost:8080", "wss://localhost:8080", "https://localhost:8080"),
        # With path
        ("wss://host/v1/listen", "wss://host/v1/listen", "https://host/v1/listen"),
        ("http://host/v1/listen", "ws://host/v1/listen", "http://host/v1/listen"),
    ],
)
def test_derive_deepgram_urls(base_url, expected_ws, expected_http):
    ws_url, http_url = _derive_deepgram_urls(base_url)
    assert ws_url == expected_ws
    assert http_url == expected_http


def test_derive_deepgram_urls_unknown_scheme_warns():
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}")
    try:
        ws_url, http_url = _derive_deepgram_urls("ftp://mydeepgram.com")
        # Falls back to secure
        assert ws_url == "wss://mydeepgram.com"
        assert http_url == "https://mydeepgram.com"
        assert "Unrecognized scheme" in sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_run_stt_send_media_exception_clears_connection():
    """send_media() failure should log a warning and clear self._connection."""
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"

    mock_connection = MagicMock()
    mock_connection.send_media = AsyncMock(side_effect=Exception("websocket closed"))
    service._connection = mock_connection

    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}")
    try:
        async for _ in service.run_stt(b"\x00" * 160):
            pass

        assert service._connection is None
        assert "send_media failed" in sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_run_stt_skips_send_when_connection_is_none():
    """When self._connection is None, run_stt should silently skip."""
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._connection = None

    # Should not raise
    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._connection is None


@pytest.mark.asyncio
async def test_connection_handler_does_not_reconnect_after_cancel():
    """A cancelled ``_connection_handler`` must die, not loop and reconnect.

    Regression test for an orphaned-reconnect zombie observed in production:
    ``_connection_handler`` is a ``while True`` reconnect loop whose
    ``finally`` block awaits ``cancel_task(keepalive_task)``. If the pipeline
    teardown cancels the connection task while it is suspended in that
    ``finally`` (e.g. right after a mid-call network drop, with the keepalive
    blocked on the dead socket), ``TaskManager.cancel_task`` used to swallow
    the handler's own ``CancelledError`` — so the loop iterated and the
    service RECONNECTED to Deepgram after having been cancelled, invisible to
    the pipeline, forever.

    The test wires a real ``TaskManager`` straight onto the service (instead
    of the full processor setup) so the genuine ``_connection_handler`` runs
    its genuine ``finally`` against the genuine ``cancel_task`` under test,
    with a fake SDK client so no network is involved.
    """
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from pipecat.utils.asyncio.task_manager import TaskManager

    task_manager = TaskManager(loop=asyncio.get_running_loop())

    service = DeepgramSTTService(api_key="fake-key-offline-test")
    service.create_task = lambda coro, name="deepgram-test": task_manager.create_task(coro, name)
    service.cancel_task = task_manager.cancel_task

    drop_event = asyncio.Event()
    connect_calls = 0

    class FakeConnection:
        def __init__(self, drops: bool):
            self._drops = drops

        def on(self, *args, **kwargs):
            pass

        async def start_listening(self):
            if self._drops:
                await drop_event.wait()
                raise ConnectionError("simulated mid-call network drop")
            await asyncio.Event().wait()  # reconnected socket: idle forever

        async def send_close_stream(self, *args, **kwargs):
            pass

        async def send_keep_alive(self, *args, **kwargs):
            pass

    def fake_connect(**kwargs):
        nonlocal connect_calls
        connect_calls += 1
        connection = FakeConnection(drops=connect_calls == 1)

        @asynccontextmanager
        async def cm():
            yield connection

        return cm()

    service._client = SimpleNamespace(
        listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect))
    )

    # Keepalive whose cancellation takes a while to complete — models the
    # real keepalive blocked mid ``send_keep_alive()`` on a just-dropped
    # socket. This holds the handler inside its finally's
    # ``await cancel_task(keepalive_task)``, the window where the race lands.
    keepalive_cancel_delivered = asyncio.Event()
    release_keepalive_cleanup = asyncio.Event()

    async def stubborn_keepalive():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            keepalive_cancel_delivered.set()
            await release_keepalive_cleanup.wait()
            raise

    service._keepalive_handler = stubborn_keepalive

    await service._connect()  # spawns the real _connection_handler
    connection_task = service._connection_task
    await asyncio.sleep(0.05)  # handler inside start_listening, keepalive parked
    assert connect_calls == 1

    # 1. The connection drops mid-call: handler enters `except Exception`,
    #    then `finally`, and suspends at `await cancel_task(keepalive_task)`.
    drop_event.set()
    await keepalive_cancel_delivered.wait()

    # 2. Pipeline teardown cancels the connection task in that exact window.
    connection_task.cancel()
    await asyncio.sleep(0.05)

    try:
        assert connection_task.cancelled() or connection_task.done(), (
            "connection handler survived an explicit cancel: its own "
            "CancelledError was swallowed inside the finally's cancel_task"
        )
        assert connect_calls == 1, (
            f"connection handler RECONNECTED after being cancelled "
            f"(connect_calls={connect_calls}) — orphaned-reconnect zombie"
        )
    finally:
        release_keepalive_cleanup.set()
        connection_task.cancel()
        await asyncio.gather(connection_task, return_exceptions=True)
        remaining = list(task_manager.current_tasks())
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
