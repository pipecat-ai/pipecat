#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the HTTP session lifecycle of FalSTTService."""

from unittest.mock import patch

import aiohttp
import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.fal.stt import FalSTTService


class _FakeResponse:
    status = 200

    async def json(self):
        return {"text": "hello there"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


async def _transcribe(service: FalSTTService) -> list:
    with patch.object(aiohttp.ClientSession, "post", lambda *args, **kwargs: _FakeResponse()):
        return [frame async for frame in service.run_stt(b"\x00\x00" * 160)]


@pytest.mark.asyncio
async def test_cleanup_closes_the_session_the_service_created():
    """A session the service opened for itself is closed at teardown."""
    service = FalSTTService(api_key="test-key")

    frames = await _transcribe(service)
    session = service._session

    assert any(isinstance(frame, TranscriptionFrame) for frame in frames)
    assert session is not None and not session.closed

    await service.cleanup()

    assert session.closed
    assert service._session is None


@pytest.mark.asyncio
async def test_cleanup_leaves_a_caller_provided_session_open():
    """A session the caller passed in belongs to the caller and stays open."""
    async with aiohttp.ClientSession() as session:
        service = FalSTTService(api_key="test-key", aiohttp_session=session)

        await _transcribe(service)
        await service.cleanup()

        assert not session.closed
        assert service._session is session


@pytest.mark.asyncio
async def test_cleanup_without_a_request_is_harmless():
    """Tearing down a service that never transcribed has no session to close."""
    service = FalSTTService(api_key="test-key")

    await service.cleanup()

    assert service._session is None
