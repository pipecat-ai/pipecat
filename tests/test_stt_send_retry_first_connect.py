#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Failed first connects in Cartesia Turns STT, which reconnects from ``send_with_retry``.

Cartesia Turns waits for the server's "connected" message before a socket
counts as ready, so it reconnects on demand when sending audio rather than from
the receive loop. A local fake provider rejects handshakes and then sends that
message on the connections it accepts.
"""

import asyncio
import json
from http import HTTPStatus

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.tests.utils import SleepFrame, run_test

FRAME_SECONDS = 0.02
MAX_RETRIES = 3


def _make_service(port: int):
    service = CartesiaTurnsSTTService(api_key="test-key", url=f"ws://127.0.0.1:{port}")
    # Keep the retry backoff short so a provider that never comes back is
    # given up on within the test.
    service._reconnect_backoff_min_wait = 0.01
    service._reconnect_backoff_max_wait = 0.01
    return service


async def _run(rejected_handshakes: int, seconds: float, confirm: bool = True):
    handshakes = 0
    audio_messages = 0

    async def process_request(connection, request):
        nonlocal handshakes
        handshakes += 1
        if handshakes <= rejected_handshakes:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable\n")
        return None

    async def handle(websocket):
        nonlocal audio_messages
        if confirm:
            connected = {"type": "connected", "request_id": "test-request"}
            await websocket.send(json.dumps(connected))
        async for message in websocket:
            if isinstance(message, bytes):
                audio_messages += 1

    async with serve(handle, "127.0.0.1", 0, process_request=process_request) as server:
        service = _make_service(server.sockets[0].getsockname()[1])
        frames = []
        for _ in range(int(seconds / FRAME_SECONDS)):
            frames.append(
                InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
            )
            frames.append(SleepFrame(sleep=FRAME_SECONDS))
        await run_test(service, frames_to_send=frames)
        # Let any connection still being served record what it received.
        await asyncio.sleep(0.05)

    return service, handshakes, audio_messages


@pytest.mark.asyncio
async def test_a_failed_first_connect_is_retried_when_audio_arrives():
    _, handshakes, audio_messages = await _run(rejected_handshakes=1, seconds=0.5)

    assert handshakes == 2
    assert audio_messages > 0


@pytest.mark.asyncio
async def test_a_provider_that_never_accepts_leaves_the_service_unusable():
    stt, handshakes, audio_messages = await _run(rejected_handshakes=100, seconds=0.5)

    # The setup connect, then one round of retries; once the service is
    # unusable it takes no more audio, so it never tries again.
    assert handshakes == 1 + MAX_RETRIES
    assert audio_messages == 0
    assert not stt.is_usable


@pytest.mark.asyncio
async def test_a_connection_that_is_never_confirmed_is_given_up_on(monkeypatch):
    # The handshake succeeds but the server never sends ``connected``, so each
    # attempt times out instead of holding the audio path forever.
    monkeypatch.setattr(CartesiaTurnsSTTService, "_CONNECTION_TIMEOUT", 0.1)

    stt, handshakes, audio_messages = await asyncio.wait_for(
        _run(rejected_handshakes=0, seconds=0.5, confirm=False), timeout=10
    )

    assert handshakes == 1 + MAX_RETRIES
    assert audio_messages == 0
    assert not stt.is_usable
