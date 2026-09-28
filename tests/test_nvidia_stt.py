#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("riva.client")

from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService
from pipecat.transcriptions.language import Language


def _make_service(**kwargs) -> NvidiaSTTService:
    return NvidiaSTTService(api_key="test-key", **kwargs)


@pytest.mark.asyncio
async def test_keepalive_enabled():
    """NVIDIA STT enables silence keepalive (the base default is off)."""
    service = _make_service()
    assert service._keepalive_timeout == 30.0
    assert service._keepalive_interval == 5.0


@pytest.mark.asyncio
async def test_keepalive_not_ready_without_iterator():
    """No active stream means keepalive should not fire."""
    service = _make_service()
    assert service._audio_iterator is None
    assert service._is_keepalive_ready() is False


@pytest.mark.asyncio
async def test_keepalive_ready_with_open_iterator():
    """An open iterator is a valid keepalive target."""
    service = _make_service()
    service._audio_iterator = AudioChunkIterator(asyncio.get_running_loop())
    assert service._is_keepalive_ready() is True


@pytest.mark.asyncio
async def test_keepalive_not_ready_with_closed_iterator():
    """A closed iterator must not be fed silence."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    await iterator.close()
    service._audio_iterator = iterator
    assert service._is_keepalive_ready() is False


@pytest.mark.asyncio
async def test_send_keepalive_enqueues_silence():
    """Silence is pushed into the active stream iterator."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    service._audio_iterator = iterator

    silence = b"\x00\x00\x00\x00"
    await service._send_keepalive(silence)

    assert iterator._queue.get_nowait() == silence


@pytest.mark.asyncio
async def test_send_keepalive_noop_when_closed():
    """Sending keepalive to a closed iterator is a no-op."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    await iterator.close()
    # close() enqueues a sentinel; drain it so the queue reflects keepalive only.
    iterator._queue.get_nowait()
    service._audio_iterator = iterator

    await service._send_keepalive(b"\x00\x00")

    assert iterator._queue.empty()


@pytest.mark.asyncio
async def test_send_keepalive_noop_without_iterator():
    """Sending keepalive with no active stream does not raise."""
    service = _make_service()
    await service._send_keepalive(b"\x00\x00")


@pytest.mark.asyncio
async def test_update_settings_reconnects_so_the_stream_uses_them(monkeypatch):
    """A settings change must reach the gRPC stream, not just the local config.

    streaming_response_generator() is handed streaming_config once, when the
    stream is opened, so rebuilding the config without reconnecting leaves the
    live stream transcribing with the previous settings and nothing logs.
    """
    service = _make_service()
    service._config = service._create_recognition_config()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(NvidiaSTTService.Settings(language=Language.ES))

    assert changed
    assert service._settings.language == Language.ES
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_settings_rebuilds_the_recognition_config(monkeypatch):
    """The rebuilt config carries the new language into the next stream."""
    service = _make_service()
    service._config = service._create_recognition_config()
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())

    assert service._config.config.language_code == Language.EN_US

    await service._update_settings(NvidiaSTTService.Settings(language=Language.ES))

    assert service._config.config.language_code == Language.ES


@pytest.mark.asyncio
async def test_update_settings_without_changes_does_not_reconnect(monkeypatch):
    """A no-op delta must not tear down a healthy stream."""
    service = _make_service()
    service._config = service._create_recognition_config()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(NvidiaSTTService.Settings())

    assert not changed
    reconnect.assert_not_awaited()
