#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Connection handling of Deepgram Flux STT against a local fake Flux server.

The service runs in a test pipeline fed with real-time 20 ms audio frames. The
fake server can reject handshakes, drop a connection, or never confirm one
with ``Connected``.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.tests.utils import SleepFrame, run_test

FRAME_SECONDS = 0.02


@dataclass
class _FakeFlux:
    rejected_handshakes: int = 0
    reject_after: int | None = None
    drop_first_after: int | None = None
    confirm: bool = True
    handshakes: int = 0
    audio_per_connection: list[int] = field(default_factory=list)

    async def process_request(self, connection, request):
        self.handshakes += 1
        rejected = self.handshakes <= self.rejected_handshakes or (
            self.reject_after is not None and self.handshakes > self.reject_after
        )
        if rejected:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable\n")
        return None

    async def handle(self, websocket):
        index = len(self.audio_per_connection)
        self.audio_per_connection.append(0)
        if self.confirm:
            await websocket.send(json.dumps({"type": "Connected", "request_id": "test"}))
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
            self.audio_per_connection[index] += 1
            if index == 0 and self.audio_per_connection[0] == self.drop_first_after:
                websocket.transport.abort()
                return


def _audio_frames(seconds: float) -> list:
    frames = []
    for _ in range(int(seconds / FRAME_SECONDS)):
        frames.append(
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
        )
        frames.append(SleepFrame(sleep=FRAME_SECONDS))
    return frames


async def _run(server: _FakeFlux, seconds: float, configure=None):
    async with serve(server.handle, "127.0.0.1", 0, process_request=server.process_request) as s:
        service = DeepgramFluxSTTService(
            api_key="test-key", url=f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}"
        )
        if configure:
            configure(service)
        down, _ = await run_test(service, frames_to_send=_audio_frames(seconds))
        # Give a connection still being served time to record what it received.
        await asyncio.sleep(0.1)
    return service, down


@pytest.mark.asyncio
async def test_a_failed_first_connect_is_retried_in_the_background():
    server = _FakeFlux(rejected_handshakes=1)

    service, _ = await _run(server, seconds=0.5)

    assert server.handshakes == 2
    assert server.audio_per_connection[0] > 0
    assert service.is_usable


@pytest.mark.asyncio
async def test_a_dropped_connection_is_replaced_by_one_new_connection():
    server = _FakeFlux(drop_first_after=10)

    await _run(server, seconds=1.0)

    assert len(server.audio_per_connection) == 2
    assert server.audio_per_connection[1] > 0


@pytest.mark.asyncio
async def test_an_unconfirmed_connection_is_given_up_for_good(monkeypatch):
    # Flux rejects an unsupported setting by never sending Connected, so a
    # retry would be rejected the same way.
    monkeypatch.setattr(DeepgramFluxSTTService, "_CONNECTION_TIMEOUT", 0.1)
    server = _FakeFlux(confirm=False)

    service, _ = await _run(server, seconds=0.5)

    assert server.handshakes == 1
    assert not service.is_usable


@pytest.mark.asyncio
async def test_audio_keeps_flowing_while_the_provider_is_down():
    # The connection drops and every reconnect is rejected. Reconnecting
    # happens in the background, so frames pass through the service on time
    # while each attempt fails and backs off.
    def long_backoff(service):
        service._reconnect_backoff_min_wait = 30
        service._reconnect_backoff_max_wait = 30

    server = _FakeFlux(drop_first_after=5, reject_after=1)

    start = time.monotonic()
    _, down = await asyncio.wait_for(_run(server, seconds=0.5, configure=long_backoff), 10)

    assert time.monotonic() - start < 3
    assert sum(isinstance(f, InputAudioRawFrame) for f in down) == 25


@pytest.mark.asyncio
async def test_stopping_does_not_reconnect():
    server = _FakeFlux()

    await _run(server, seconds=0.3)
    await asyncio.sleep(0.3)

    assert server.handshakes == 1
