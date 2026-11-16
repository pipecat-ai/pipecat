#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reconnect behavior of websocket STT services against a local fake provider.

Each service runs in a test pipeline fed with real-time 20 ms audio frames. The
fake provider answers every few audio messages with a final transcript naming
the connection that received them, so a transcript reaches downstream only when
the socket receiving audio is the socket the receive loop reads.
"""

import asyncio
import json
from dataclasses import dataclass, field
from http import HTTPStatus

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.smallest.stt import SmallestSTTService
from pipecat.tests.utils import SleepFrame, run_test

SERVICES = ["cartesia", "elevenlabs", "smallest"]

# Long enough that audio frames keep arriving while a handshake is in flight.
HANDSHAKE_DELAY = 0.1
TRANSCRIPT_EVERY = 5
FRAME_SECONDS = 0.02


@dataclass
class _Connection:
    id: int
    audio_messages: int = 0


@dataclass
class _FakeProvider:
    service: str
    drop_first_after: int | None = None
    reject_first_handshake: bool = False
    connections: list[_Connection] = field(default_factory=list)
    handshakes: int = 0

    async def process_request(self, connection, request):
        self.handshakes += 1
        await asyncio.sleep(HANDSHAKE_DELAY)
        if self.reject_first_handshake and self.handshakes == 1:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable\n")
        return None

    def _is_audio(self, message) -> bool:
        if self.service == "elevenlabs":
            return json.loads(message).get("message_type") == "input_audio_chunk"
        return isinstance(message, bytes)

    def _transcript(self, text: str) -> str:
        if self.service == "cartesia":
            return json.dumps({"type": "transcript", "text": text, "is_final": True})
        if self.service == "elevenlabs":
            return json.dumps({"message_type": "committed_transcript", "text": text})
        return json.dumps({"transcript": text, "is_final": True})

    async def handle(self, websocket):
        conn = _Connection(id=len(self.connections) + 1)
        self.connections.append(conn)
        if self.service == "elevenlabs":
            await websocket.send(json.dumps({"message_type": "session_started"}))
        async for message in websocket:
            if not self._is_audio(message):
                continue
            conn.audio_messages += 1
            if conn.id == 1 and conn.audio_messages == self.drop_first_after:
                websocket.transport.abort()
                return
            if conn.audio_messages % TRANSCRIPT_EVERY == 0:
                await websocket.send(self._transcript(f"c{conn.id}"))


def _make_service(name: str, port: int):
    if name == "cartesia":
        return CartesiaSTTService(api_key="test-key", base_url=f"ws://127.0.0.1:{port}")
    if name == "smallest":
        return SmallestSTTService(api_key="test-key", base_url=f"ws://127.0.0.1:{port}")
    service = ElevenLabsRealtimeSTTService(api_key="test-key", base_url=f"127.0.0.1:{port}")
    websocket_connect = service._websocket_connect

    async def plain_websocket_connect(uri, **kwargs):
        return await websocket_connect(uri.replace("wss://", "ws://", 1), **kwargs)

    service._websocket_connect = plain_websocket_connect
    return service


async def _run(provider: _FakeProvider, seconds: float) -> list[str]:
    async with serve(
        provider.handle, "127.0.0.1", 0, process_request=provider.process_request
    ) as server:
        port = server.sockets[0].getsockname()[1]
        frames = []
        for _ in range(int(seconds / FRAME_SECONDS)):
            frames.append(
                InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
            )
            frames.append(SleepFrame(sleep=FRAME_SECONDS))
        down, _ = await run_test(_make_service(provider.service, port), frames_to_send=frames)
    return [f.text for f in down if isinstance(f, TranscriptionFrame)]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_dropped_connection_is_replaced_by_one_new_connection(service):
    provider = _FakeProvider(service, drop_first_after=15)

    transcripts = await _run(provider, seconds=1.5)

    assert len(provider.connections) == 2
    second = provider.connections[1]
    assert second.audio_messages >= TRANSCRIPT_EVERY
    # Allow for a transcript still in flight when the pipeline ends.
    assert transcripts.count("c2") >= second.audio_messages // TRANSCRIPT_EVERY - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_failed_first_connect_is_retried(service):
    provider = _FakeProvider(service, reject_first_handshake=True)

    transcripts = await _run(provider, seconds=1.0)

    assert provider.handshakes == 2
    assert len(provider.connections) == 1
    assert "c1" in transcripts
