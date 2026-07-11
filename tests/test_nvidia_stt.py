#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import pytest

pytest.importorskip("riva.client")

from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService


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
