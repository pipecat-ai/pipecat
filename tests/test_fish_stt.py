#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.fish.stt import (
    FishAudioSTTService,
    FishAudioSTTSettings,
    language_to_fish_language,
)
from pipecat.transcriptions.language import Language


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload or {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, response=None):
        self._response = response or _FakeResponse(payload={"text": "hello", "language_code": "en"})
        self.url = None
        self.headers = None
        self.form = None

    def post(self, url, *, data, headers):
        self.url = url
        self.headers = headers
        self.form = data
        return self._response


def _form_fields(form):
    return {field[0]["name"]: field[2] for field in form._fields}


@pytest.mark.asyncio
async def test_fish_stt_uploads_segment_as_multipart():
    session = _FakeSession()
    service = FishAudioSTTService(
        api_key="fish_test",
        aiohttp_session=session,
        settings=FishAudioSTTSettings(language=Language.JA),
    )

    result = await service._transcribe_audio(b"RIFF")

    assert result["text"] == "hello"
    assert session.url == "https://api.fish.audio/v1/asr"
    assert session.headers["Authorization"] == "Bearer fish_test"
    fields = _form_fields(session.form)
    assert fields["language"] == "ja"
    assert fields["ignore_timestamps"] == "true"


@pytest.mark.asyncio
async def test_fish_stt_yields_transcription_frame():
    session = _FakeSession(
        _FakeResponse(payload={"text": "  hello there  ", "language_code": "en"})
    )
    service = FishAudioSTTService(api_key="fish_test", aiohttp_session=session)

    frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "hello there"
    assert frames[0].language == "en"


@pytest.mark.asyncio
async def test_fish_stt_skips_empty_transcription():
    session = _FakeSession(_FakeResponse(payload={"text": "   "}))
    service = FishAudioSTTService(api_key="fish_test", aiohttp_session=session)

    frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert frames == []


@pytest.mark.asyncio
async def test_fish_stt_yields_error_frame_on_api_failure():
    session = _FakeSession(_FakeResponse(status=402, text="insufficient credit"))
    service = FishAudioSTTService(api_key="fish_test", aiohttp_session=session)

    frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "402" in frames[0].error


@pytest.mark.asyncio
async def test_fish_stt_does_not_close_a_borrowed_session():
    session = _FakeSession()
    service = FishAudioSTTService(api_key="fish_test", aiohttp_session=session)

    assert not service._owns_session

    await service.cleanup()

    assert service._session is session


def test_language_to_fish_language_drops_region():
    assert language_to_fish_language(Language.EN_US) == "en"
    assert language_to_fish_language(Language.TA) == "ta"
