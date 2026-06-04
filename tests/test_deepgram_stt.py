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
