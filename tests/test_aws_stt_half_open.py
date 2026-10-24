#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A half-open AWS Transcribe socket must be detected and replaced.

A local websocket server stands in for AWS. On the first connection it stops
reading and writing without closing, so the peer looks alive but never answers
(including websocket pings). The service must notice and reconnect, and the new
connection must receive audio.
"""

import asyncio

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.tests.utils import SleepFrame, run_test

FRAME_SECONDS = 0.02
# Keepalive timing the test injects where the service leaves the library default
# (20 s) in place, so detection takes well under a second.
TEST_PING_INTERVAL = 0.2
TEST_PING_TIMEOUT = 0.2


class _HalfOpenProvider:
    def __init__(self):
        self.connections = 0
        self.stuck = None
        self.second_connection_audio = 0

    async def handle(self, websocket):
        self.connections += 1
        if self.connections == 1:
            # Peer vanished without a close frame: stop answering anything.
            self.stuck = websocket
            websocket.transport.pause_reading()
            await websocket.wait_closed()
            return
        async for message in websocket:
            if isinstance(message, bytes):
                self.second_connection_audio += 1


def _make_service(port: int) -> AWSTranscribeSTTService:
    service = AWSTranscribeSTTService(
        api_key="test-secret",
        aws_access_key_id="test-key",
        region="us-east-1",
    )
    websocket_connect = service._websocket_connect

    async def local_websocket_connect(uri, **kwargs):
        # The service adds handshake headers of its own; a standards-checking local
        # server rejects the duplicate Sec-WebSocket-* ones, which AWS tolerates.
        headers = kwargs.get("additional_headers") or {}
        kwargs["additional_headers"] = {
            k: v for k, v in headers.items() if not k.startswith("Sec-WebSocket-")
        }
        # Only fill in the ping timing when the service did not choose its own.
        # A service that passes ping_interval=None keeps it disabled.
        kwargs.setdefault("ping_interval", TEST_PING_INTERVAL)
        kwargs.setdefault("ping_timeout", TEST_PING_TIMEOUT)
        return await websocket_connect(f"ws://127.0.0.1:{port}/", **kwargs)

    service._websocket_connect = local_websocket_connect
    return service


@pytest.mark.asyncio
async def test_a_half_open_socket_is_replaced_by_a_new_connection():
    provider = _HalfOpenProvider()

    async with serve(provider.handle, "127.0.0.1", 0, subprotocols=["mqtt"]) as server:
        port = server.sockets[0].getsockname()[1]
        frames = []
        for _ in range(int(3.0 / FRAME_SECONDS)):
            frames.append(
                InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
            )
            frames.append(SleepFrame(sleep=FRAME_SECONDS))
        try:
            await run_test(_make_service(port), frames_to_send=frames)
        finally:
            # A socket that never reads cannot finish a close handshake.
            if provider.stuck is not None:
                provider.stuck.transport.abort()

    assert provider.connections == 2
    assert provider.second_connection_audio > 0
