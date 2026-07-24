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
async def test_run_stt_enqueues_audio():
    """run_stt() should put audio into the queue and yield None."""
    import asyncio

    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._audio_queue = asyncio.Queue()

    audio = b"\x00" * 160
    results = [frame async for frame in service.run_stt(audio)]

    assert results == [None]
    assert service._audio_queue.get_nowait() == audio


@pytest.mark.asyncio
async def test_audio_sender_clears_connection_on_exception():
    """send_media() failure should log a warning and clear self._connection."""
    import asyncio
    import contextlib

    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"
    service._audio_queue = asyncio.Queue()

    mock_connection = MagicMock()
    mock_connection.send_media = AsyncMock(side_effect=Exception("websocket closed"))
    service._connection = mock_connection
    service._connection_ready = asyncio.Event()
    service._connection_ready.set()

    await service._audio_queue.put(b"\x00" * 160)

    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}")
    try:
        task = asyncio.create_task(service._audio_sender_handler())
        await asyncio.sleep(0)  # let the sender dequeue the item and call send_media()
        await asyncio.sleep(0)  # let send_media() raise, handler catches and clears _connection
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert service._connection is None
        assert "send_media failed" in sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_audio_sender_skips_send_when_connection_is_none():
    """When self._connection is None, the audio sender should consume the chunk without calling send_media."""
    import asyncio
    import contextlib

    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._connection = None
    service._audio_queue = asyncio.Queue()
    service._connection_ready = asyncio.Event()

    await service._audio_queue.put(b"\x00" * 160)

    task = asyncio.create_task(service._audio_sender_handler())
    await asyncio.sleep(0)  # let sender dequeue and drop the chunk
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Chunk was consumed from the queue (not stuck), no send_media call made
    assert service._audio_queue.empty()


if __name__ == "__main__":
    unittest.main()
