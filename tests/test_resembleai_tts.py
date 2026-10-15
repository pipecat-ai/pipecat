#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import pytest
from websockets.asyncio.server import serve

from pipecat.services.resembleai.tts import ResembleAITTSService


@pytest.mark.asyncio
async def test_resembleai_receive_returns_when_server_closes():
    """A clean server close ends the receive loop so the base class can reconnect."""

    async def handler(ws):
        await ws.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        service = ResembleAITTSService(
            api_key="test-key", voice_id="test-voice", url=f"ws://127.0.0.1:{port}"
        )
        await service._connect_websocket()

        await asyncio.wait_for(service._receive_messages(), timeout=2)

        await service._disconnect_websocket()
