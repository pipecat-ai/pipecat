#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS text normalization."""

import pytest

from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame
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


def _service(session, **settings) -> MiniMaxHttpTTSService:
    return MiniMaxHttpTTSService(
        api_key="key",
        group_id="group",
        aiohttp_session=session,
        settings=MiniMaxHttpTTSService.Settings(**settings),
    )


async def _request(**settings) -> dict:
    session = _RecordingSession()
    await run_test(_service(session, **settings), frames_to_send=[TTSSpeakFrame("Hello.")])
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


def test_text_normalization_setting_is_deprecated():
    with pytest.warns(DeprecationWarning, match="Settings.text_normalization"):
        _service(_RecordingSession(), text_normalization=True)


@pytest.mark.asyncio
async def test_text_normalization_update_is_deprecated():
    session = _RecordingSession()
    with pytest.warns(DeprecationWarning, match="Settings.text_normalization"):
        await run_test(
            _service(session),
            frames_to_send=[
                TTSUpdateSettingsFrame(
                    delta=MiniMaxHttpTTSService.Settings(text_normalization=True)
                ),
                TTSSpeakFrame("Hello."),
            ],
        )
    assert session.payloads[0]["text_normalization_mode"] == "basic"
