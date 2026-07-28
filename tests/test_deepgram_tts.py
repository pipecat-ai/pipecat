#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Deepgram TTS error handling."""

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.status import ServiceStatus


def _websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


@pytest.mark.asyncio
async def test_deepgram_rejected_api_key_misconfigures_the_service(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr("pipecat.services.deepgram.tts.websocket_connect", fake_websocket_connect)

    service = DeepgramTTSService(api_key="wrong-key", sample_rate=24000)

    await service._connect_websocket()

    assert service.status == ServiceStatus.MISCONFIGURED


@pytest.mark.asyncio
async def test_deepgram_unavailable_service_stays_usable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(503)

    monkeypatch.setattr("pipecat.services.deepgram.tts.websocket_connect", fake_websocket_connect)

    service = DeepgramTTSService(api_key="test-key", sample_rate=24000)

    await service._connect_websocket()

    assert service.status.is_usable
