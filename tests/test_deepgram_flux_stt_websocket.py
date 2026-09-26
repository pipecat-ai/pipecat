#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Flux WebSocket recovery when a peer stops accepting writes."""

import asyncio
import socket
from unittest.mock import AsyncMock, Mock, call

import pytest
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 5.0])
async def test_healthy_sends_preserve_audio_and_control_messages(timeout):
    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=timeout)
    websocket = AsyncMock(state=State.OPEN)
    service._websocket = websocket

    await service.send_with_retry(b"audio", service._report_error)
    await service._transport_send_json({"type": "CloseStream"})

    assert websocket.send.await_args_list == [call(b"audio"), call('{"type": "CloseStream"}')]
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_stalled_audio_send_reconnects_and_retries():
    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=0.01)
    service._sample_rate = 8000
    stalled = AsyncMock(state=State.OPEN)

    async def stall(message):
        await asyncio.Event().wait()

    def abort():
        stalled.state = State.CLOSED

    stalled.send.side_effect = stall
    stalled.transport = Mock()
    stalled.transport.abort.side_effect = abort
    healthy = AsyncMock(state=State.OPEN)
    service._websocket = stalled

    async def connect():
        service._websocket = healthy

    service._connect_websocket = AsyncMock(side_effect=connect)
    audio = b"\0" * 320

    async def transcribe():
        return [frame async for frame in service.run_stt(audio)]

    assert await asyncio.wait_for(transcribe(), timeout=1) == [None]
    service._connect_websocket.assert_awaited_once()
    stalled.send.assert_awaited_once_with(audio)
    assert stalled.state is State.CLOSED
    healthy.send.assert_awaited_once_with(audio)


@pytest.mark.asyncio
async def test_stalled_retry_yields_error_instead_of_hanging():
    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=0.01)
    service._sample_rate = 8000
    first = AsyncMock(state=State.OPEN)
    first.send.side_effect = ConnectionError("connection lost")
    retry = AsyncMock(state=State.OPEN)
    retry.transport = Mock()

    async def stall(message):
        await asyncio.Event().wait()

    async def reconnect(**kwargs):
        service._websocket = retry
        return True

    retry.send.side_effect = stall
    service._websocket = first
    service._try_reconnect = AsyncMock(side_effect=reconnect)

    async def transcribe():
        return [frame async for frame in service.run_stt(b"\0" * 320)]

    frames = await asyncio.wait_for(transcribe(), timeout=1)
    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "websocket send timed out" in frames[0].error
    retry.transport.abort.assert_called_once()
    retry.wait_closed.assert_awaited_once()
    service._try_reconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["silence", "configure", "disconnect"])
async def test_stalled_control_or_silence_send_closes_connection(operation):
    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=0.01)
    service._sample_rate = 8000
    websocket = AsyncMock(state=State.OPEN)

    async def stall(message):
        await asyncio.Event().wait()

    def abort():
        websocket.state = State.CLOSED

    websocket.send.side_effect = stall
    websocket.transport = Mock()
    websocket.transport.abort.side_effect = abort
    service._websocket = websocket
    service.push_error = AsyncMock()

    async with asyncio.timeout(1):
        if operation == "disconnect":
            await service._disconnect_websocket()
            assert service._websocket is None
            service.push_error.assert_awaited_once()
        else:
            with pytest.raises(TimeoutError, match="websocket send timed out"):
                if operation == "silence":
                    await service._send_silence()
                else:
                    await service._send_configure({"eot_threshold"})

    assert websocket.state is State.CLOSED
    websocket.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_canceling_send_does_not_reconnect():
    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=5)
    websocket = AsyncMock(state=State.OPEN)
    websocket.transport = Mock()
    started = asyncio.Event()

    async def stall(message):
        started.set()
        await asyncio.Event().wait()

    websocket.send.side_effect = stall
    service._websocket = websocket
    service._try_reconnect = AsyncMock()
    task = asyncio.create_task(service.send_with_retry(b"audio", service._report_error))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    service._try_reconnect.assert_not_awaited()
    websocket.close.assert_not_awaited()
    websocket.transport.abort.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_peer_that_stops_reading_is_bounded():
    """Fill a real TCP write buffer; neither send nor the close can drain."""
    ready = asyncio.Event()
    release_peer = asyncio.Event()

    async def deaf_peer(websocket):
        websocket.transport.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, 4096
        )
        websocket.transport.pause_reading()
        ready.set()
        try:
            await release_peer.wait()
        finally:
            websocket.transport.abort()

    service = DeepgramFluxSTTService(api_key="test", ws_send_timeout=0.05)
    async with serve(deaf_peer, "127.0.0.1", 0, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        websocket = await service._websocket_connect(
            f"ws://127.0.0.1:{port}",
            compression=None,
            ping_interval=None,
            close_timeout=0.05,
        )
        service._websocket = websocket
        websocket.transport.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, 4096
        )
        try:
            await asyncio.wait_for(ready.wait(), timeout=1)
            async with asyncio.timeout(3):
                with pytest.raises(TimeoutError, match="websocket send timed out"):
                    # TCP buffer sizes vary by OS; keep writing until flow
                    # control suspends a send rather than assuming one fills it.
                    while True:
                        await service._transport_send_audio(b"\0" * (1024 * 1024))
            assert websocket.state is State.CLOSED
        finally:
            release_peer.set()
            await websocket.close()


@pytest.mark.parametrize("timeout", [0, -1])
def test_send_timeout_must_be_positive(timeout):
    with pytest.raises(ValueError, match="ws_send_timeout must be positive"):
        DeepgramFluxSTTService(api_key="test", ws_send_timeout=timeout)
