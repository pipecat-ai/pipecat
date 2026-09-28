#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the WebSocket client transport."""

import asyncio
from unittest.mock import AsyncMock

import pytest
import websockets

import pipecat.transports.websocket.client as websocket_client
from pipecat.frames.frames import EndFrame, ErrorFrame, Frame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.client import (
    WebsocketClientCallbacks,
    WebsocketClientInputTransport,
    WebsocketClientOutputTransport,
    WebsocketClientParams,
    WebsocketClientSession,
    WebsocketClientTransport,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory


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


@pytest.mark.asyncio
async def test_a_connect_timeout_is_reported_by_both_transports(monkeypatch):
    """A handshake that times out is reported like any other failure to connect.

    The input and output transports share the connection, so each reports that
    it could not be set up.
    """

    async def timing_out_connect(**kwargs):
        raise TimeoutError("timed out during opening handshake")

    monkeypatch.setattr(websocket_client, "websocket_connect", timing_out_connect)

    transport = WebsocketClientTransport(uri="ws://example.com")
    worker = PipelineWorker(Pipeline([transport.input(), transport.output()]))
    errors: list[ErrorFrame] = []

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        errors.append(frame)

    await worker.queue_frame(EndFrame())
    async with asyncio.timeout(5):
        await worker.run(WorkerParams(task_manager=TaskManager()))

    assert len(errors) == 2
    assert {type(error.processor) for error in errors} == {
        WebsocketClientInputTransport,
        WebsocketClientOutputTransport,
    }
    assert all(error.category == ErrorCategory.CONNECTIVITY for error in errors)
    assert all(isinstance(error.exception, TimeoutError) for error in errors)


class _CoalescingSerializer(FrameSerializer):
    """Emits one coalesced payload every third frame, buffering the two before it."""

    def __init__(self):
        super().__init__()
        self._seen = 0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Emit the accumulated block on every third audio frame."""
        if not isinstance(frame, OutputAudioRawFrame):
            return None
        self._seen += 1
        if self._seen % 3:
            return None
        return frame.audio * 3

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Unused; only the output transport is exercised here."""
        return None


@pytest.mark.asyncio
async def test_every_frame_is_paced_when_payloads_are_coalesced():
    """Tests for issue #5592.

    A serializer that buffers audio across calls emits no payload on most of
    them. Those frames have still been written, so pacing follows the frames
    taken rather than the payloads that go out.
    """
    params = WebsocketClientParams(serializer=_CoalescingSerializer(), audio_out_enabled=True)
    output = WebsocketClientTransport(uri="ws://localhost:1", params=params).output()
    output._sample_rate = 16000
    output._write_audio_sleep = AsyncMock()

    connection = AsyncMock()
    connection.state = websockets.State.OPEN
    output._session._websocket = connection

    frame = OutputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
    written = [await output.write_audio_frame(frame) for _ in range(9)]

    assert written == [True] * 9
    assert output._write_audio_sleep.await_count == 9
    assert connection.send.await_count == 3
