#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS text normalization."""

import pytest

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.minimax.tts import MiniMaxHttpTTSService
from pipecat.tests.utils import run_test


class _RejectingResponse:
    """A response that fails, so a request is recorded without audio."""

    status = 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingSession:
    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return _RejectingResponse()


async def _request(**settings) -> dict:
    session = _RecordingSession()
    service = MiniMaxHttpTTSService(
        api_key="key",
        group_id="group",
        aiohttp_session=session,
        settings=MiniMaxHttpTTSService.Settings(**settings),
    )
    await run_test(service, frames_to_send=[TTSSpeakFrame("Hello.")])
    return session.payloads[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings, mode",
    [
        ({}, None),
        ({"text_normalization_mode": "quality"}, "quality"),
        ({"text_normalization": True}, "basic"),
        ({"text_normalization": False}, None),
        ({"text_normalization": True, "text_normalization_mode": "quality"}, "quality"),
    ],
)
async def test_request_sends_the_text_normalization_mode(settings, mode):
    request = await _request(**settings)
    assert request.get("text_normalization_mode") == mode
    assert "text_normalization" not in request["voice_setting"]
